from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.document_loaders import DataFrameLoader

import os
import fitz
import pandas as pd
import json

def fonts(doc, granularity=False, pages=2):
    """Extracts fonts and their usage in PDF documents.

    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool

    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}
    pageCounter = 0

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage
        pageCounter += 1
        if pageCounter >= pages:
            break

    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, styles


def font_tags(font_counts, styles):
    """Returns dictionary with font sizes as keys and tags as value.

    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict

    :rtype: dict
    :return: all element tags based on font-sizes
    """
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)

    return size_tag


def headers_para(doc, size_tag, pages=2):
    """Scrapes headers & paragraphs from PDF and return texts with element tags.

    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict

    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = []  # list with headers and paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span

    pageCounter = 0
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # this block contains text

                # REMEMBER: multiple fonts and sizes are possible IN one block

                block_string = ""  # text found in block
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if s['text'].strip():  # removing whitespaces:
                            if first:
                                previous_s = s
                                first = False
                                block_string = size_tag[s['size']] + s['text']
                            else:
                                if s['size'] == previous_s['size']:

                                    if block_string and all((c == "|") for c in block_string):
                                        # block_string only contains pipes
                                        block_string = size_tag[s['size']] + s['text']
                                    if block_string == "":
                                        # new block has started, so append size tag
                                        block_string = size_tag[s['size']] + s['text']
                                    else:  # in the same block, so concatenate strings
                                        block_string += " " + s['text']

                                else:
                                    header_para.append(block_string)
                                    block_string = size_tag[s['size']] + s['text']

                                previous_s = s

                    # new block started, indicating with a pipe
                    block_string += "|"

                header_para.append(block_string)
        pageCounter += 1
        if pageCounter >= pages:
            break

    return header_para


def get_pdf_first_page_txt(pdf_path, pages=2):
    doc = fitz.open(pdf_path)

    font_counts, styles = fonts(doc, granularity=False, pages=pages)

    size_tag = font_tags(font_counts, styles)

    return headers_para(doc, size_tag, pages)


def get_pdf_page_metadata(pdf_path, pages):
    pdf_first_page_txt = get_pdf_first_page_txt(pdf_path, pages)

    template = """
                I have a paragraph which was extracted from the first page of a Journal of Economic Literature (JEL) PDF file. 
                The paragraph typically begins with the word 'Abstract' and there is usually a 'Keywords' section following it. 
                I would like you to extract and return the article title, author, abstract section, and keywords section.
                If you come across JEL classifications such as C12 and P34, please disregard them and do not include them in the abstract or keywords.

                {format_instructions}

                Wrap your final output as a json objects

                INPUT:
                {pdf_first_page_txt}

                YOUR RESPONSE:
    """
    response_schemas = [
        ResponseSchema(name="title", description="extracted title"),
        ResponseSchema(name="author", description="extracted authors seperated by comma"),
        ResponseSchema(name="abstract", description="extracted abstract"),
        ResponseSchema(name="keywords", description="extracted keywords")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(template)  
        ],
        input_variables=["pdf_first_page_txt"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0.0, max_tokens=2048) # type: ignore

    final_prompt = prompt.format_prompt(pdf_first_page_txt=pdf_first_page_txt)
    output = llm(final_prompt.to_messages())

    try:
        result = output_parser.parse(output.content)
    except:
        if "```json" in output.content:
            json_string = output.content.split("```json")[1].strip()
        else:
            json_string = output.content
        result = json.loads(json_string)

    head, tail = os.path.split(pdf_path)

    result["filename"] = tail

    return result


def save_pdfs_metadata_to_db(pdf_files, excel_file, pages=1):
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
        existing_data = df.to_dict(orient='records')
    else:
        existing_data = []

    existing_filenames = set(row['filename'] for row in existing_data)

    
    for doc in pdf_files:
        head, tail = os.path.split(doc)
        if tail not in existing_filenames:
            print('get meta from LLM '+doc)
            try:
                metadata = get_pdf_page_metadata(doc, pages)
                temp_data = []
                temp_data.append(metadata)
                save_to_excel(existing_data+temp_data, excel_file)
                existing_data += temp_data
                print("Data append to ", excel_file)
            except Exception as e:
                print(str(e))


def get_metadata_from_db(excel_file):
    df = pd.read_excel(excel_file)
    dict = df.to_dict(orient='records',)
    return dict


def get_column_from_db(excel_file, column):
    df = pd.read_excel(excel_file)
    doc = DataFrameLoader(df, column).load()
    return doc


def save_to_excel(data, file_path):
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    
def main():
    
    documents = ['./data/docs/abstracts/tujula2007.pdf']    
    output_file = "data/db/repo.xlsx"
    save_pdfs_metadata_to_db(documents, output_file, 1)

if __name__ == '__main__':
    main()