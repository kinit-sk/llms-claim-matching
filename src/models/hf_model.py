from typing import List, Union
from src.models.model import Model
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFModel(Model):
    """
    A class to represent a Hugging Face model.
    
    Attributes:
        name (str): The name of the model.
        max_new_tokens (int): The maximum number of tokens to generate.
        do_sample (bool): Whether to use sampling.
        device_map (str): The device map.
        load_in_4bit (bool): Whether to load the model in 4-bit.
        load_in_8bit (bool): Whether to load the model in 8-bit.
        offload_folder (str): The offload folder.
        offload_state_dict (bool): Whether to offload the state dict.
        max_memory (Any): The maximum memory to use.
        system_prompt (str): The system prompt to use.    
    """
    def __init__(
        self, 
        name: str = 'google/gemma-7b-it',
        max_new_tokens: int = 128, 
        do_sample: bool = False, 
        device_map: str = 'auto', 
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs
    ):
        super().__init__(name='HFModel', max_new_tokens=max_new_tokens)
        self.model_name = name
        self.tokenizer = None
        self.model = None
        self.device_map = eval(device_map) if '{' in device_map else device_map
        self.do_sample = do_sample
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        if 'offload_folder' not in kwargs:
            self.offload_folder = None
            self.offload_state_dict = None
        else:
            self.offload_folder = kwargs['offload_folder']
            self.offload_state_dict = kwargs['offload_state_dict']
            
        if 'max_memory' not in kwargs:
            self.max_memory = None
        else:
            self.max_memory = eval(kwargs['max_memory']) if kwargs['max_memory'] else None
        self.system_prompt = kwargs['system_prompt'] if kwargs['system_prompt'] != 'None' else None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load()
        
    def _load_model(self) -> None:
        """
        Load the model.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map = self.device_map,
            offload_folder = self.offload_folder,
            offload_state_dict = self.offload_state_dict,
            max_memory = self.max_memory
        )
        
    def load_quantized_model(self) -> None:
        """
        Load the quantized model.
        """
        print(f'Loading quantized model - {self.model_name}')
        if 'gemma-2-27b-it' in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.bfloat16,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                load_in_8bit=self.load_in_8bit,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                device_map=self.device_map, 
                quantization_config=quantization_config
            )

    def load(self) -> 'HFModel':
        """
        Load the Hugging Face model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.load_in_8bit or self.load_in_4bit:
            self.load_quantized_model()
        else:
            self._load_model()

        logging.log(
            logging.INFO, f'Loaded model and tokenizer from {self.model_name}')
        
    def _is_chat(self) -> bool:
        """
        Check if the model is a chat model.
        """
        # if 'gemma-2' in self.model_name:
        #     return False
        return hasattr(self.tokenizer, 'chat_template')
    
    def _get_system_role(self) -> str:
        """
        Get the system role.
        """
        if 'gemma' in self.model_name or 'mistral' in self.model_name:
            return None
        else:
            return 'system'
    
    def _terminators(self) -> List[int]:
        """
        Get the terminators.
        """
        if 'Llama-3' in self.model_name:
            return [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            return [
                self.tokenizer.eos_token_id
            ]
            
    def _get_answer_probs(self, logits):         
        last_token_logits = logits[0]
        
        yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("No")
        probabilities = F.softmax(last_token_logits, dim=-1)
        yes_prob = probabilities[0, yes_token_id].item()
        no_prob = probabilities[0, no_token_id].item()
        
        return yes_prob, no_prob

    def infere(self, prompt: Union[str, List[str]]) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt (Union[str, List[str]]): The prompt to generate text from.
            
        Returns:
            str: The generated text.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        
        answers = []
        yes_probs = []
        no_probs = []
        for p in prompt:
            if self._is_chat():
                if self.system_prompt and self._get_system_role() == 'system':
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": p}
                    ]
                elif self.system_prompt:
                    messages = [
                        {"role": "user", "content": f'{self.system_prompt}\n\n{p}'}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": p}
                    ]
                
                inputs = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True,
                    return_tensors='pt'
                ).to(self.device)

            else:
                if 'falcon-40b' in self.model_name:
                    if self.system_prompt is not None:
                        p = f'{self.system_prompt}\nUser: {p}\nFalcon:'
                    else:
                        p = f'User: {p}\nFalcon:'
                    
                elif self.system_prompt is not None:
                    p = f'{self.system_prompt}\n\n{p}'

                inputs = self.tokenizer(
                    p, 
                    return_tensors='pt'
                ).to(self.device)['input_ids']
            
            generated_output = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self._terminators(),
                return_dict_in_generate=True,
                output_logits=True,
                do_sample=self.do_sample,
            )
            generated_ids = generated_output.sequences
            logits = generated_output.logits
            
            yes_prob, no_prob = self._get_answer_probs(logits)
            yes_probs.append(yes_prob)
            no_probs.append(no_prob)
            
            decoded_input = self.tokenizer.batch_decode(
                inputs, 
                skip_special_tokens=True
            )[0]
            
            decoded = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            decoded = decoded[len(decoded_input):]
            answers.append(decoded.strip())
        
        return '##### '.join(answers), yes_probs, no_probs
