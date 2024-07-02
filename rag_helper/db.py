from lancedb.pydantic import LanceModel
from itertools import batched
from tqdm import tqdm

def get_table(db, table_name: str, schema: LanceModel | None = None):
    if table_name in db.table_names():
        return db.open_table(table_name)
    
    if schema is None:
        raise ValueError(
            f"Table {table_name} does not exist and no schema was provided"
        )
    
    return db.create_table(table_name, schema=schema, mode="overwrite")

def insert_data_into_table(table, data, batch_size=20):
    batches = batched(data, batch_size)

    for batch in tqdm(batches):
        table.add(list(batch))