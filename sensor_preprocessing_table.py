from gpt4all import GPT4All
import os, sys, time
from datetime import datetime
import re


def generate_prompt(sensor, sensor_list):
    new_list = sensor_list[:]
    new_list.remove(sensor)
    new_sensor_string = ', '.join(new_list)
    prompt = "What are some unique image preprocessing techniques for images captured by {0} sensors in computer vision. Only list techniques that are relevant to formatting and do not include image enhancement or noise reduction. List unique techniques in context which the preprocessing techniques are not used for images captured by other visual sensors such as {1}.".format(
        sensor, new_sensor_string)
    return prompt


def append_response_to_file(response, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(response + '\n\n')


if __name__ == '__main__':
    model_folder = 'C:/Users/{0}/OneDrive - James Cook University/Models/LLM'.format(os.getlogin())
    m_name = 'wizard'
    if m_name == 'falcon':
        model_name = 'ggml-model-gpt4all-falcon-q4_0.bin'
        model_name = os.path.join(model_folder, model_name)
    elif m_name == 'wizard':
        model_name = 'wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin'
        model_name = os.path.join(model_folder, model_name)
    else:
        model_name = None

    sensor_list = ['RGB', 'Multispectral', 'Hyperspectral', 'NIR', 'IR', 'Thermographic', 'Ultrasonic', 'Depth Camera',
                   'LiDAR']

    current_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    file_path = '{0}_responses_{1}.txt'.format(m_name, current_time)
    table_file_path = 'table_' + file_path

    if not os.path.isfile(file_path):
        open(file_path, 'w').close()  # Create an empty file if it doesn't exist
    if not os.path.isfile(table_file_path):
        open(file_path, 'w').close()  # Create an empty file if it doesn't exist

    all_responses = list()
    max_tokens = 1000

    if m_name is not None:
        model = GPT4All(model_name=model_name)
    else:
        model = GPT4All('ggml-model-gpt4all-falcon-q4_0.bin')

    with model.chat_session():

        for i in range(len(sensor_list)):
            print("Sensor: {0}".format(sensor_list[i]))

            prompt = generate_prompt(sensor_list[i], sensor_list)
            response = model.generate(prompt, max_tokens=max_tokens)
            print("Response: {0}".format(response))
            append_response_to_file(response, file_path)
            all_responses.append(response)

            prompt = "Generate a LaTeX table with the headings Technique, Purpose and Description using the information from the following list: {0}".format(
                all_responses[i])
            response = model.generate(prompt, max_tokens=max_tokens)
            print("Response: {0}".format(response))
            append_response_to_file(response, table_file_path)

    time.sleep(60)
    os.system('shutdown -s')
