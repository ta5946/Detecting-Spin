import os
import dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# TODO langchain

dotenv.load_dotenv()
HF_SECRET = os.environ.get('HF_SECRET')
HF_MODEL = 'allenai/OLMo-7B-hf'
N_TOKENS = 4

model = AutoModelForCausalLM.from_pretrained(HF_MODEL,
                                             token=HF_SECRET,
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL,
                                          token=HF_SECRET,
                                          trust_remote_code=True)
pipe = pipeline('text-generation',
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=N_TOKENS)

input = """
    Question: Does the RCT sentence report results in test group not relative to control group?
    Sentence: All patients entered the hospital were not previously under any medication.
    Answer: No
    
    Question: Does the RCT sentence report results in test group not relative to control group?
    Sentence: Baby wipes had an equivalent effect on skin hydration when compared with cotton wool and water .
    Answer: No
    
    Question: Does the RCT sentence report results in test group not relative to control group?
    Sentence: Maternal albendazole tended to reduce these effects.
    Answer: Yes
    
    Question: Does the RCT sentence report results in test group not relative to control group?
    Sentence: All patients improved on CGI, GAF, and STAXI scores after 6 and 12 months, independently of treatment received.
    Answer: Yes
    
    Question: Does the RCT sentence report results in test group not relative to control group?
    Sentence: No significant differences were observed between the two treatment groups at baseline.
"""

output = pipe(input)
print(output[0]['generated_text'])
