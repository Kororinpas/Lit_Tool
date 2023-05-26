def get_chain_output(query,docs):
    
    # docs = vectordb.similarity_search(query,5,include_metadata=True)
    
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # type: ignore
    
    from langchain.prompts import PromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
    from langchain.llms import OpenAI
    
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field, validator
    from typing import List,Union,Optional
    
    class Sentence(BaseModel):
        sentence: List[str] = Field(description="The sentence in the given document which is the most similar to the query provided")
        source: List[str] = Field(description="The meta source of the paper")
        score: List[float] = Field(description = "The similarity score between the sentence selected and the query provided")
            
    parser = PydanticOutputParser(pydantic_object=Sentence)
    
    question_template = """
    Given the document and query, find three sentences in the document that are most similar in meaning to the query. 
    Return the sentences, the meta source of the sentences and the cosine similarity scores. 
    If no similar sentences is found, return the sentence with highest cosine siliarity scores.
    {query}
    ===========
    {context}
    ===========
    {format_instructions}
    
    """

    from langchain.chains.question_answering import load_qa_chain
    from langchain import LLMChain

    PROMPT = PromptTemplate(template = question_template,
                            input_variables=['query','context'],
                            partial_variables = {"format_instructions":parser.get_format_instructions()})
    
    llm_chain = LLMChain(llm=llm,prompt = PROMPT)
    
    output = llm_chain({"query":query,"context":docs})
    
    return output
    