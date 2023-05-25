from cosine_match import search_cosine_similarity

def run_text_match(output,query,docs,embeddings):
    
    import re
    text = re.sub("\n+","",output['text'])
    
    import json
    json_obj = json.loads(text)
    
    
    if "properties" in json_obj:
        print('No result was found, Using embedding searching strategy!!!')
        split_docs = _split_for_embedding(docs)
        similar_sentence = search_cosine_similarity(query,split_docs,embeddings)
    
        for i,element in enumerate(similar_sentence):
            print(f'The {i} sentence')
            print(f"Sentence:{element['sentences']}")
            print(f"Source:{element['source']}")
            print(f"Score:{element['score']}")
            print("========.")
            print("========.")
    else:
        for i in range(3):
            print(f'The {i} sentence')
            print(f"Sentence:{json_obj['sentence'][i]}")
            print(f"Source:{json_obj['source'][i]}")
            print(f"Score:{json_obj['score'][i]}")
            print("========_")
            print("========_")

def _split_for_embedding(docs): ##输入docs(list),输出split_for embedding(list)
    for_embedding = []
    for content in docs:
        new_content = content.page_content.replace('et al.','et al。')
        new_content = new_content.split('.')
        
        meta_data = content.metadata
        
        for split_content in new_content:
            split_content = split_content.replace('。','.')
            
            if len(split_content) < 30:
                continue
            else:
                for_embedding.append({"content":split_content,"source":meta_data})
                
    return for_embedding