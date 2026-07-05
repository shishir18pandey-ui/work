this is the final payload which you will receive.
Look at fileContent, it has been finalised that fileContent will always come as base64 encoded strings, not as binary files.
You can take upon the development accordingly.
 
In /create API you will receive this API.
then that file which is being passed, check filetype, as we will only entertain image for now.
pass fileContent to VL model and extract text and add it to incident description.


        





        openai:
  vl_base_url: "https://qwen3-vl-8b.iservebetter.idfcfirstbank.com/v1"
  vl_model_name: "/app/models/Qwen3-VL-8B-Instruct"




  - name: VISION_API_BASE
  value: "{{ .Values.openai.vl_base_url }}"
- name: VISION_MODEL_NAME
  value: "{{ .Values.openai.vl_model_name }}"
