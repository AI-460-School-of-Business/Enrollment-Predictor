"""
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model designed to understand the context of words in a sentence by considering both the left and right context simultaneously. 
It's typically used for natural language processing tasks such as text classification and named entity recognition. 
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from db_connectors import PostgresConnector


tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-6B")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-6B")

test_string = "What are all the courses offered in the School of Business at CCSU?"

postgres_connector = PostgresConnector(
    user=USER, password=PASSWORD, dbname=DATABASE, host=HOST, port=PORT
)

postgres_connector.connect()
db_schema = [postgres_connector.get_schema(table) for table in postgres_connector.get_tables()]
formatter = RajkumarFormatter(db_schema)

manifest_client = Manifest(client_name="huggingface", client_connection="http://127.0.0.1:5000")

def get_sql(instruction: str, max_tokens: int = 300) -> str:
    prompt = formatter.format_prompt(instruction)
    res = manifest_client.run(prompt, max_tokens=max_tokens)
    return formatter.format_model_output(res)

print(get_sql("Number of rows in table?"))