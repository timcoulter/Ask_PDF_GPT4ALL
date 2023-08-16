import os, re, tqdm
from ask_pdfs import process_bibliography
from gpt4all import GPT4All
from datetime import datetime


storage = r'C:\Users\Tim\OneDrive - James Cook University\Zotero Storage'
bib = r'C:\Users\Tim\Documents\Zotero_bibs\Surveys.bib'

def check_substring(string, substr_list):
    for substr in substr_list:
        if substr in string:
            return True
    return False

# Get the current directory
current_directory = os.getcwd()

# Get a list of all files in the current directory
all_files = os.listdir(current_directory)

# Filter the files to include only those whose filenames contain 'question_output'
question_output_files = [file for file in all_files if 'question_output' in file]

#get titles and pdf filepaths of valid files
print("Processing bibliography")
title, file = process_bibliography(bib, storage)

articles = []
new_titles = []

for i in range(len(question_output_files)):
    with open(question_output_files[i], 'r', encoding='utf-8') as f:
        article_response = ''
        
        for line in f:
            result = check_substring(line, title)
            x = line.strip() 
            if result and article_response == '':
                new_titles.append(x)
                article_response = ''
            elif result and article_response != '':
                new_titles.append(x)
                articles.append(article_response)
                article_response = ''
            elif x != '':
                article_response = article_response + x + ' '
            
        articles.append(article_response)
    
for i in range(len(articles)):
    print(new_titles[i])
    print(articles[i])
    print()
    
print(len(new_titles))
print(len(articles))


                
model_path = r'C:\Users\Tim\OneDrive - James Cook University\Models\LLM\wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin'
#model_path = r'C:\Users\Tim\OneDrive - James Cook University\Models\LLM\ggml-model-gpt4all-falcon-q4_0.bin'
model = GPT4All(model=model_path, max_tokens=1000, n_predict=1000, verbose=False, repeat_last_n=0)
#model = GPT4All(model_name=model_path, max_tokens=1000, n_predict=1000, verbose=False, repeat_last_n=0)
#model = GPT4All(model_name=model_path)

current_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
file_path = 'finetuned_{}.txt'.format(current_time)

for i in tqdm.tqdm(range(len(articles))):
    print(new_titles[i])
    prompt = 'Return a comma seperated list of challenges. Use the following text : {0}'.format(articles[i])
    with model.chat_session():
        response = model.generate(prompt)
        print(response)
        
    
