import math
import pandas as pd
import json
import logging

from area.evapi import EngineeringVillageAPI

logger = logging.getLogger(__name__)


class SearchEV(EngineeringVillageAPI):
    def __init__(
        self,
        apikey=None,
        insttoken=None,
        startyear=None,
        endyear=None,
        updateNumber=None,
    ):
        """Main object used for searching engineering village using the
        https://api.elsevier.com/content/ev/results	route.
        See https://dev.elsevier.com/documentation/EngineeringVillageAPI.wadl

        Parameters
        ----------
        apikey : str, optional
            API key used for querying EV REST, by default None
        insttoken : str, optional
            insitutional token used in REST request, by default None
        startyear : int, optional
            year to start searching, by default None
        endyear : int, optional
            year to end searching if nothing chosen it will search until latest, by default None
        """
        EngineeringVillageAPI.__init__(self, apikey=apikey, insttoken=insttoken)
        self._startyear = startyear
        self._endyear = endyear
        self.updateNumber = None
        if updateNumber:
            if updateNumber >= 1 and updateNumber <= 4:
                self.updateNumber = updateNumber

    def _check_year(self, year: int):
        """Check the year entered is actually valid

        Parameters
        ----------
        year : int
            input year

        Returns
        -------
        int, bool
            output year when true, otherwise none and false
        """
        try:
            year = int(year)
        except ValueError:
            logger.error(f"Cannot {year} as a year, invalid type")
            return None, False

        if year > 0:
            digits = int(math.log10(year)) + 1
        elif year == 0:
            digits = 1
        else:
            logger.error("Year cannot be negative")
            return None, False
        if digits == 4:
            return year, True
        else:
            logger.error("Year should have four digits")
            return None, False

    @property
    def startyear(self):
        return self._startyear

    @startyear.setter
    def startyear(self, year):
        year, state = self._check_year(year)
        if state:
            self._startyear = year

    @property
    def endyear(self):
        return self._endyear

    @endyear.setter
    def endyear(self, year):
        year, state = self._check_year(year)
        if state:
            self._endyear = year

    def _build_search(
        self,
        query: str,
        pagesize: int,
        offset: int,
        autoStemming=None,
        updateNumber=None,
        language=None,
        document_type=None,
    ):
        """Helper to assemble to search url

        Parameters
        ----------
        query : str
            the word to query
        pagesize : int
            number of results in page (max 100)
        offset : int
            page offset, e.g. offset=(N-1)*pagesize = page N
        autoStemming : int, optional
            _description_, by default None
        updateNumber : int, optional
            results within the last updateNumber of weeks, by default None
        language : str, optional
            string with language e.g. English, by default None
        document_type : str, optional
            string showing document type search phrase
            {ca} OR {ja} - conference or journal, by default None

        Returns
        -------
        str
            search url
        """
        # build a advanced query term by adding language and document type filters if required
        query = f'(("{query}") WN ALL)'
        if language:
            query = f"({query} AND ({language} WN LA))"
        if document_type:
            query = f"(({query} AND (({document_type}) WN DT)))"
        url = (
            self.base_url
            + f"/results?query={query}&pageSize={pagesize}&offset={offset*pagesize}"
        )
        if self.updateNumber:
            url += f"&updateNumber={self.updateNumber}"
        if self.startyear:
            url += f"&startYear={self.startyear}"
        if self.endyear:
            url += f"&endYear={self.endyear}"
        return url

    def _parse_results(self, results):
        """Function to extract what we care about from the search results
        Example results https://api.engineeringvillage.com/EvDataWebServices/ev/results.json
        Currently, we just keep the document id and use this to retrieve the record from EV

        Parameters
        ----------
        results : list
            list of dictionaries extracted from the search page entry

        Returns
        -------
        list
            list containing the parsed results
        """
        output = []
        for result in results:
            output.append([result["EI-DOCUMENT"]["DOC"]["DOC-ID"]])
        return output

    def get_document(self, docId):
        """Function to retrieve the records using EV rest api

        Parameters
        ----------
        docId : str
            document id for EV

        Returns
        -------
        dict
            EI-Document
        """
        query_url = self.base_url + f"/records?docId={docId}"
        resp = self._get_request(query_url)
        if resp.status_code == 200:
            return json.loads(resp.text)["PAGE"]["PAGE-RESULTS"]["PAGE-ENTRY"][0][
                "EI-DOCUMENT"
            ]
        return {"error": docId}

    def search(
        self,
        query,
        nresults=None,
        pagesize=100,
        language="English",
        document_type="{ca} OR {ja}",
    ):
        """Main entry point for searching engineering village

        Parameters
        ----------
        query : str
            word/phrase to query
        nresults : int, optional
            total number of results, if None will return all, by default None
        pagesize : int, optional
            number of results per page max 100, by default 100
        language : str, optional
            string with language e.g. English, by default "English"
        document_type : str, optional
            type of documents to return, by default "{ca} OR {ja}"

        Returns
        -------
        list
            list of document ids
        """
        search_results = []
        if nresults and nresults < pagesize:
            pagesize = nresults
        # initial query to find how many results there are, offset 0 means take the first pagesize queries
        page_offset = 0
        while True:
            query_url = self._build_search(
                query,
                pagesize,
                page_offset,
                language=language,
                document_type=document_type,
            )
            try:
                resp = self._get_request(query_url)
                results = json.loads(resp.text)
                if not nresults:
                    nresults = results["PAGE"]["RESULTS-COUNT"]
                # try:
                if results["PAGE"]["RESULTS-COUNT"] == 0:
                    print("no results")
                    return search_results, False
                search_results.extend(
                    self._parse_results(results["PAGE"]["PAGE-RESULTS"]["PAGE-ENTRY"])
                )
            except BaseException as e:
                pass
                print(f"Failed to search for: {query_url} with error {e}")

            # if resp.status_code != 200:
            #     return search_results, False

            print(
                f"Total number of results {nresults}, current page offset {page_offset}, current results length {len(search_results)}"
            )
            nresults -= pagesize
            page_offset += 1
            if nresults <= 0:
                return search_results, True
