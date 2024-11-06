Certainly! I'll provide a complete, end-to-end example that covers the entire process from CSV merging to hierarchical chunking and FAISS index creation. This script will be a single unit that you can run (assuming you have the necessary CSV files and AWS credentials set up).

```python
import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
import boto3
import json

def join_csv_files(file_paths, join_condition):
    dfs = [pd.read_csv(file) for file in file_paths]
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=join_condition, how='left')
    return merged_df

def hierarchical_chunk_dataframe(df, category_column, text_columns):
    documents = []

    for category, group in df.groupby(category_column):
        content_parts = []
        
        for _, row in group.iterrows():
            item_dict = row.drop(category_column).to_dict()
            item_dict = {k: str(v) for k, v in item_dict.items()}
            item_str = json.dumps(item_dict, ensure_ascii=False)
            content_parts.append(item_str)
        
        full_content = " ".join(content_parts)
        
        doc = Document(
            page_content=full_content,
            metadata={category_column: category}
        )
        documents.append(doc)

    return documents

def create_faiss_index(chunks):
    boto3.setup_default_session(region_name='us-west-2')  # Replace with your AWS region

    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=boto3.client('bedrock-runtime')
    )

    faiss_index = FAISS.from_documents(chunks, bedrock_embeddings)
    return faiss_index

# Main execution
if __name__ == "__main__":
    # Step 1: Merge CSV files
    file_paths = ['file1.csv', 'file2.csv', 'file3.csv']
    join_condition = 'id'
    merged_df = join_csv_files(file_paths, join_condition)

    print("Merged DataFrame:")
    print(merged_df)
    print()

    # Step 2: Hierarchical Chunking
    category_column = 'category'
    text_columns = ['subcategory', 'product', 'description', 'content']
    chunks = hierarchical_chunk_dataframe(merged_df, category_column, text_columns)

    print("Chunks created:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content length: {len(chunk.page_content)} characters")
        print(f"Content preview: {chunk.page_content[:250]}...")
        print(f"Number of records: {chunk.page_content.count('{')}")
        print()

    # Step 3: Create FAISS index
    faiss_index = create_faiss_index(chunks)
    faiss_index.save_local("my_category_faiss_index")
    print("Category-based FAISS index created and saved successfully.")

    # Optional: Demonstrate a simple search
    query = "electronics"
    results = faiss_index.similarity_search(query, k=1)
    print(f"\nSearch Results for '{query}':")
    for doc in results:
        print(f"Category: {doc.metadata['category']}")
        print(f"Content preview: {doc.page_content[:100]}...")
```

To use this script:

1. Ensure you have the necessary CSV files (`file1.csv`, `file2.csv`, `file3.csv`) in the same directory as the script.
2. Install required packages: `pip install pandas langchain faiss-cpu boto3`
3. Set up your AWS credentials for Bedrock access.
4. Run the script.

This script will:

1. Merge the CSV files based on the 'id' column.
2. Create hierarchical chunks where each chunk represents a category and contains all items in that category.
3. Create a FAISS index using these chunks and Bedrock embeddings.
4. Save the FAISS index locally.
5. Perform a simple search to demonstrate the functionality.

Example output (assuming the same data as in previous examples):

```
Merged DataFrame:
   id    category subcategory      product         description                                   content
0   1 Electronics  Smartphones   iPhone 12  Latest iPhone model  The iPhone 12 features a powerful A14...
1   1 Electronics     Laptops MacBook Air  Lightweight laptop   The MacBook Air is incredibly thin an...
2   2       Books     Fiction    The Great Gatsby  Classic novel  F. Scott Fitzgerald's The Great Gats...
3   3    Clothing    T-Shirts Cotton T-Shirt Comfortable wear   Our cotton T-shirt is made from 100% ...

Chunks created:
Chunk 1:
Metadata: {'category': 'Electronics'}
Content length: 523 characters
Content preview: {"id": "1", "subcategory": "Smartphones", "product": "iPhone 12", "description": "Latest iPhone model", "content": "The iPhone 12 features a powerful A14 Bionic chip, 5G capability, and a stunning Super Retina XDR display..."} {"id": "1", "subcategory": "Laptops", "product": "MacBook Air", ...
Number of records: 2

Chunk 2:
Metadata: {'category': 'Books'}
Content length: 261 characters
Content preview: {"id": "2", "subcategory": "Fiction", "product": "The Great Gatsby", "description": "Classic novel", "content": "F. Scott Fitzgerald's The Great Gatsby is a tragic love story, a mystery, and a social commentary on American life..."}
Number of records: 1

Chunk 3:
Metadata: {'category': 'Clothing'}
Content length: 258 characters
Content preview: {"id": "3", "subcategory": "T-Shirts", "product": "Cotton T-Shirt", "description": "Comfortable wear", "content": "Our cotton T-shirt is made from 100% organic cotton, providing ultimate comfort and breathability..."}
Number of records: 1

Category-based FAISS index created and saved successfully.

Search Results for 'electronics':
Category: Electronics
Content preview: {"id": "1", "subcategory": "Smartphones", "product": "iPhone 12", "description": "Latest iPhone model"...
```

This script provides a complete workflow from data merging to search-ready vector database creation, with hierarchical chunking based on categories. You can adjust the `category_column` and `text_columns` variables to fit your specific data structure.