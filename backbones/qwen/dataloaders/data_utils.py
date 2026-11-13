"""One image preprocessing pass, answer-safe labels, and explicit visual budgets."""
import copy
import numpy as np
import torch
from PIL import Image, ImageOps

SESAME_SYSTEM_PROMPT = ('Check whether the requested target is visible. Reply briefly. '
    'Say present or absent, name an alternative if known, and emit [SEG] only for a present target that should be segmented.')


def configure_vision_budget(processor, min_pixels=4096, max_pixels=1048576):
    if not 0 < min_pixels <= max_pixels:
        raise ValueError('Invalid pixel budget')
    ip = processor.image_processor
    ip.min_pixels, ip.max_pixels = min_pixels,max_pixels
    if hasattr(ip,'size') and isinstance(ip.size,dict):
        ip.size = {'shortest_edge':min_pixels,'longest_edge':max_pixels}


class Qwen3VLImageProcessor:
    def __init__(self,processor,image_size=1024,min_pixels=None,max_pixels=None):
        self.processor,self.image_size = processor,image_size
        ip = getattr(processor,'image_processor',None)
        size = getattr(ip,'size',{}) or {}
        self.min_pixels = min_pixels or size.get('shortest_edge',4096)
        self.max_pixels = max_pixels or size.get('longest_edge',1048576)
        if ip is not None:
            configure_vision_budget(processor,self.min_pixels,self.max_pixels)

    def process_image(self,image_path,process_qwen=True,prepare_sam=True):
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        sam_image,size = self._prepare_sam_input(image) if prepare_sam else (None,None)
        inputs = self.processor(text='',images=[image],return_tensors='pt',
            min_pixels=self.min_pixels,max_pixels=self.max_pixels) if process_qwen else {}
        return {'image':image,'sam_image':sam_image,'sam_input_size':size,
                'original_size':(image.height,image.width),'resized_size':size,
                'pixel_values':inputs.get('pixel_values'),'image_grid_thw':inputs.get('image_grid_thw')}

    def _prepare_sam_input(self,image):
        scale = self.image_size/max(image.height,image.width)
        h,w = int(image.height*scale+.5),int(image.width*scale+.5)
        resized = image.resize((w,h),Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(resized,dtype=np.float32)).permute(2,0,1).contiguous()
        return tensor[None],(h,w)

    def process_for_demo(self,image_path):
        r = self.process_image(image_path)
        return r['sam_image'],r['pixel_values'],r['original_size']


def _expanded_prompt_ids(processor,prompt,grids):
    ids = processor.tokenizer(prompt,add_special_tokens=False)['input_ids']
    image_id = processor.tokenizer.convert_tokens_to_ids(getattr(processor,'image_token','<|image_pad|>'))
    merge = processor.image_processor.merge_size
    counts = [] if grids is None else [int(g.prod())//(merge*merge) for g in grids]
    result,index = [],0
    for token in ids:
        if token==image_id:
            if index>=len(counts):
                raise ValueError('Image placeholders do not match image grids')
            result.extend([token]*counts[index]);index += 1
        else:
            result.append(token)
    if index!=len(counts):
        raise ValueError('Prompt is missing an image placeholder')
    return torch.tensor(result,dtype=torch.long)


def encode_conversation(processor,messages,image,max_length=2048):
    messages = copy.deepcopy(messages)
    for message in messages:
        if isinstance(message.get('content'),list):
            for block in message['content']:
                if block.get('type')=='image':
                    block['image'] = image
    positions = [i for i,m in enumerate(messages) if m['role']=='assistant']
    if not positions:
        raise ValueError('Training sample has no assistant answer')
    prefix = messages[:positions[-1]]
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    prompt = processor.apply_chat_template(prefix,tokenize=False,add_generation_prompt=True)
    # Pixels are processed only once. The prefix's image-token expansion is exact.
    ip = processor.image_processor
    tokens = processor(text=text,images=[image],return_tensors='pt',truncation=False,
                       min_pixels=ip.min_pixels,max_pixels=ip.max_pixels)
    ids = tokens['input_ids'][0]
    prompt_ids = _expanded_prompt_ids(processor,prompt,tokens.get('image_grid_thw'))
    n = len(prompt_ids)
    if n>=max_length or not torch.equal(ids[:n].cpu(),prompt_ids):
        raise ValueError('Prompt budget exceeded or assistant boundary does not match; no image-token truncation is allowed.')
    ids = ids[:max_length]
    labels = ids.clone();labels[:n] = -100
    answer=messages[positions[-1]].get('content','')
    if isinstance(answer,str) and '[SEG]' in answer:
        marker=processor.tokenizer.convert_tokens_to_ids('[SEG]')
        if not (labels==marker).any():
            raise ValueError('The required assistant SEG marker was truncated')
    tokens.update(input_ids=ids,labels=labels,attention_mask=torch.ones_like(ids))
    return tokens


def encode_prompt(processor,messages,image,max_length=2048):
    """Use the same serialization path as training for mixed text/image messages."""
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ip=processor.image_processor
    tokens=processor(text=text,images=[image],return_tensors='pt',truncation=False,
                     min_pixels=ip.min_pixels,max_pixels=ip.max_pixels)
    if tokens['input_ids'].shape[1]>max_length:
        raise ValueError('Image/text budget exceeded; image-token truncation is forbidden')
    return tokens


def collate_fn_train(batch,tokenizer=None,processor=None,max_length=2048,**kwargs):
    batch = [member for item in batch for member in (item if isinstance(item,list) else [item])]
    if not batch or processor is None:
        raise ValueError('Nonempty batch and multimodal processor are required')
    items = [encode_conversation(processor,item['conversations'],item['image'],max_length) for item in batch]
    tokenizer = processor.tokenizer
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    length = max(len(item['input_ids']) for item in items)
    def padded(key,value):
        return torch.stack([torch.cat([item[key],item[key].new_full((length-len(item[key]),),value)]) for item in items])
    pixels = [item['pixel_values'] for item in items if item.get('pixel_values') is not None]
    grids = [item['image_grid_thw'] for item in items if item.get('image_grid_thw') is not None]
    return {'input_ids':padded('input_ids',pad),'labels':padded('labels',-100),
            'attention_mask':padded('attention_mask',0),
            'pixel_values':torch.cat(pixels) if pixels else None,
            'image_grid_thw':torch.cat(grids) if grids else None,
            'sam_images':[item['sam_image'] for item in batch],
            'image_keys':[item.get('image_path') for item in batch],
            'sam_mask_shape_list':[item['sam_mask_shape'] for item in batch],
            'masks_list':[item.get('masks') for item in batch],
            'ignore_masks_list':[item.get('ignore_mask') for item in batch],
            'exists_list':[item.get('exists') for item in batch],
            'pair_ids':[item.get('pair_id') for item in batch],
            'pair_kinds':[item.get('pair_kind') for item in batch]}


def collate_fn_val(batch,**kwargs):
    return collate_fn_train(batch,**kwargs)


def find_seg_token_positions(input_ids,seg_token_idx):
    return input_ids==seg_token_idx
