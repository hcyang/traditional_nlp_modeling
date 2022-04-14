
export APP_MODULE_PATH=wikipedia_processor
export APP_MODULE_FILE_NAME=app_wikipedia_dump_xml_processor

export WIKIPEDIA_DUMP_INPUT_PATH=~/data_wikipedia/dumps.wikimedia.org_zhwiki_20220401
export WIKIPEDIA_DUMP_XML_FILE_NAME=zhwiki-20220401-pages-articles.xml.bz2
export WIKIPEDIA_DUMP_OUTPUT_PATH=~/data_wikipedia/dumps.wikimedia.org_zhwiki_20220401_output

python -m $APP_MODULE_PATH.$APP_MODULE_FILE_NAME --input_path=$WIKIPEDIA_DUMP_INPUT_PATH --file_name=$WIKIPEDIA_DUMP_XML_FILE_NAME --output_path=$WIKIPEDIA_DUMP_OUTPUT_PATH
