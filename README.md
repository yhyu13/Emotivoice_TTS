
# Install instructions

Assume you are in the root directory of the text-generation-webui

```
cd ./extensions
git clone --depth 1 --recursive https://github.com/yhyu13/Emotivoice_TTS.git

cd ./Emotivoice_TTS
pip install -r requirements.txt

cd ./EmotiVoice
git lfs pull
pip install -r requirements.txt
git lfs clone --depth 1 https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese
rm -rf outputs
git lfs clone --depth 1 https://www.modelscope.cn/syq163/outputs.git

cd ../../../
python sever.py --extensions EmotiVoice_TTS
```