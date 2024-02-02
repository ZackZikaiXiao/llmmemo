nohup python memorize.py > memorize.out 2>&1 &
175430


取消peft：
1. parse.py中peft改为false
2. llmmemo/model_utils/get_model.py中所有load_in_8bit从False改成True
3. memorize.py中，TrainingArguments中，fp16从True改成False
4. evaluate.py中，line94改成False；注释掉line98