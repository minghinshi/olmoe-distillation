cd ./src
python -c "import utils; utils.save_tokenizer()"
cd ..
cp ./chat_template.jinja ./models/tokenizer/chat_template.jinja
