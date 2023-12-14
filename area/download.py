import concurrent.futures
import json
from .document import Document
import tqdm
import pandas as pd


def download_documents(document_ids: list) -> pd.DataFrame:
    """Iterate through a list of document ids and download

    Parameters
    ----------
    document_ids : list
        ev document ids

    Returns
    -------
    pd.DataFrame
        results table
    """

    output = []
    for docid in document_ids:
        try:
            output.append(get_document(docid))
        except Exception as e:
            print(e)
            print(f"Cannot retrieve {docid}")

    return pd.DataFrame(
        output,
        columns=[
            "Title",
            "Citation",
            "DOI",
            "Abstract",
            "Classification_index_terms",
            "Uncontrolled_terms",
            "Controlled_terms",
            "Year",
            "Document_type",
            "Journal_name",
            "Authors",
            "Authors_affiliation",
            "prop",
            "obj",
        ],
    )


def parallel_download_documents(document_ids):
    """Iterate through a list of document ids and download
    in parallel

    Parameters
    ----------
    document_ids : list
        a list of document id strings

    Returns
    -------
    list, list
        list of results e.g. [[row1],[row2]], headings
    """
    output = []
    # run multiple downloads in parallel to avoid slow api calls
    # this means Engineering village and doi.org are called once each per
    # thread
    with concurrent.futures.ThreadPoolExecutor() as e:
        fut = [e.submit(get_document, docid) for docid in document_ids]
        for r in tqdm.tqdm(concurrent.futures.as_completed(fut)):
            try:
                output.append(r.result())
            except Exception as e:
                print(e)
                # print(f"Cannot retrieve {docid}")
    return pd.DataFrame(
        output,
        columns=[
            "Title",
            "Citation",
            "DOI",
            "Abstract",
            "Classification_index_terms",
            "Uncontrolled_terms",
            "Controlled_terms",
            "Year",
            "Document_type",
            "Journal_name",
            "Authors",
            "Authors_affiliation",
            "prop",
            "obj",
        ],
    )


def get_document(docId: str) -> list:
    """Helper function for calling during parallel code

    Parameters
    ----------
    docId : str
        ev document id

    Returns
    -------
    list
        row of results table
    """
    document = Document(docId)
    return [
        document.title,
        document.citation,
        document.doi,
        document.abstract,
        document.classification_indexed_terms,
        document.uncontrolled_terms,
        document.controlled_terms,
        document.year,
        document.document_type,
        document.journal_name,
        document.authors,
        document.affiliations,
        json.dumps(document.properties),
        json.dumps(document.objects),
    ]
