from gpt4all import GPT4All
import os
from datetime import datetime
import re
import time


def add_period(sentence):
    # Regular expression pattern to match end of sentence punctuation
    pattern = r'[.!?]$'

    if not re.search(pattern, sentence):
        sentence += '.'

    return sentence


def append_response_to_file(response, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(response + '\n\n')


if __name__ == '__main__':

    model_path = r'C:\Users\{0}\AppData\Local\nomic.ai\GPT4All\ggml-model-gpt4all-falcon-q4_0.bin'.format(os.getlogin())
    groovy_path = r'C:\Users\{0}\AppData\Local\nomic.ai\GPT4All\ggml-gpt4all-j-v1.3-groovy.bin'.format(os.getlogin())
    hermes_path = r'C:\Users\{0}\AppData\Local\nomic.ai\GPT4All\gnous-hermes-13b.ggmlv3.q4_0.bin'.format(os.getlogin())
    wizard_path = r'C:\Users\{0}\OneDrive - James Cook University\Models\LLM\wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin'.format(os.getlogin())

    model = GPT4All(model_name=wizard_path, n_threads=os.cpu_count()/4)
    #model.model.set_thread_count(os.cpu_count())



    max_tokens = 1000
    temp = 2
    top_k = 40
    top_p = 0.1
    repeat_penalty = 1.18
    repeat_last_n = 1000
    n_batch= os.cpu_count()
    n_predict = max_tokens
    streaming = False

    greedy_k = 1
    greedy_p = 1

    start_prompt = 'Continue the detailed story, do not conclude it, from the following: '
    first_sentence = 'Once a upon a time in a magical land there was a character named Tim, he lived in an old forest.'
    print(first_sentence)

    current_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    file_path = 'responses_{}.txt'.format(current_time)

    if not os.path.isfile(file_path):
        open(file_path, 'w').close()  # Create an empty file if it doesn't exist

    with model.chat_session():
        idx = 1



        response = model.generate(start_prompt + first_sentence, max_tokens=max_tokens, temp=temp, top_k=top_k,
                                  top_p=top_p,
                                  repeat_penalty=repeat_penalty, repeat_last_n=repeat_last_n, n_batch=n_batch,
                                  n_predict=n_predict,
                                  streaming=streaming)

        response = add_period(response)
        print('{0}: {1}'.format(idx, response))
        append_response_to_file(response, file_path)

        all_responses = []
        all_responses.append(first_sentence)
        all_responses.append(response)

        valid = True
        while valid:
            idx += 1

            start = time.time()

            if idx % 2 == 0:

                new_response = model.generate(start_prompt + response, max_tokens=max_tokens, temp=temp, top_k=top_k,
                                              top_p=top_p,
                                              repeat_penalty=repeat_penalty, repeat_last_n=repeat_last_n, n_batch=n_batch,
                                              n_predict=n_predict,
                                              streaming=streaming)
            else:

                new_response = model.generate(start_prompt + response, max_tokens=max_tokens, temp=temp, top_k=top_k,
                                              top_p=top_p,
                                              repeat_penalty=repeat_penalty, repeat_last_n=repeat_last_n,
                                              n_batch=512,
                                              n_predict=n_predict,
                                              streaming=streaming)

            end = time.time()
            diff = end - start


            new_response = add_period(new_response)
            print('{0}: Time: {1}, Response: {2}'.format(idx,diff,new_response))
            append_response_to_file(new_response, file_path)

            all_responses.append(new_response)
            if new_response == response:
                valid = False
            else:
                response = new_response

        print("THE END")
        append_response_to_file("THE END", file_path)

    print("Responses saved to:", file_path)
