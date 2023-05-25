def get_pdf_metadata(path):
    metadata = 0
    info = _get_pdf_info(path)

    if '/doi' in info:
        doi = info['/doi']
    elif '/Subject' in info:
        Subject = info['/Subject']
        if 'doi:' in Subject:
            Subject = Subject.split('doi:')
            doi = Subject[1]
        else:
          metadata = 'None'
    elif '/WPS-ARTICLEDOI' in info:
        doi = info['/WPS-ARTICLEDOI']
    else:
        metadata = 'None'
    
    if metadata != 'None':
        import habanero
        import time
        citation = habanero.cn.content_negotiation(ids = doi,format='bibentry')
        time.sleep(5)
        import bibtexparser
        citation = bibtexparser.loads(citation)
        citation = citation.entries[0]
        metadata = {'author':citation['author'],
              'year':citation['year'],
              'title':citation['title'],
              'journal':citation['journal'],
              }
    
    return metadata

def _get_pdf_info(path):
    from PyPDF2 import PdfReader
    with open(path, 'rb') as f:
        pdf = PdfReader(f)
        info = pdf.metadata
        return info