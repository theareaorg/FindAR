import re
import requests
import os
import json

from area.evapi import EngineeringVillageAPI

CLEANR = re.compile("<.*?>")  # regex for cleaning html


class Document(EngineeringVillageAPI):
    def __init__(self, docId, apikey=None, insttoken=None):
        """Class for downloading and accessing documents from
        https://api.elsevier.com/content/ev/records

        Parameters
        ----------
        docId : str
            engineering village document id string
        """
        EngineeringVillageAPI.__init__(self, apikey=apikey, insttoken=insttoken)
        resp = self._get_request(self._build_retrival(docId))
        eidocument = json.loads(resp.text)["PAGE"]["PAGE-RESULTS"]["PAGE-ENTRY"][0][
            "EI-DOCUMENT"
        ]
        self.properties = eidocument["DOCUMENTPROPERTIES"]
        self.objects = eidocument["DOCUMENTOBJECTS"]
        self.aus = eidocument.get("AUS", {"AU": []})  # skip if no authors
        self.afs = eidocument.get("AFS", {"AF": []})  # skip if no affiliations
        self.bibstyle = "apa"

    def _build_retrival(self, docId) -> str:
        return self.base_url + f"/records?docId={docId}"

    @property
    def doi(self):
        return self.properties.get("DO", "")

    @property
    def citation(self):
        """Call doi.org api to get the citation of the paper

        Returns
        -------
        _type_
            _description_
        """
        try:
            r = requests.get(
                f"https://doi.org/{self.doi}",
                headers={"Accept": "text/x-bibliography", "style": f"{self.bibstyle}"},
            )
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            # sometimes doi.org returns a html page saying the doi is not found, rather than an invalid
            # status code. So we just check the text, <html> tag is in it or if its too long don't use it
            if "<html>" in r.text or len(r.text) > 500:
                return None
            return r.text
        except:
            return None

    @property
    def abstract(self):
        # clean html tags from the abstract
        return re.sub(CLEANR, "", self.properties.get("AB", ""))

    @property
    def classification_indexed_terms(self):
        all_cl = self.objects.get("CLS", {}).get("CL", {})
        terms = []
        for c in all_cl:
            code = c["CID"]
            description = c["CTI"]
            terms.append(f"{code} {description}")
        return ";".join(terms)

    @property
    def uncontrolled_terms(self):
        return ";".join(self.objects.get("FLS", {}).get("FL", ""))

    @property
    def controlled_terms(self):
        return ";".join(self.objects.get("CVS", {}).get("CV", ""))

    @property
    def year(self):
        if "YR" in self.properties:
            return self.properties["YR"]
        if "PD_YR" in self.properties:
            return self.properties["PD_YR"]
        else:
            return ""
        # return self.properties.get('YR','')

    @property
    def document_type(self):
        return self.properties.get("DT", "")

    @property
    def journal_name(self):
        if "SE" in self.properties:
            return self.properties["SE"]
        if "SO" in self.properties:
            return self.properties["SO"]
        else:
            return ""
        # return self.properties.get('SE','')

    @property
    def title(self):
        return self.properties.get("TI", "")

    @property
    def authors(self):
        authors = ""
        for author in self.aus["AU"]:
            aid = author.get("ID", None)
            name = author["NAME"]
            authors += f"({aid}) {name}; "
        return authors

    @property
    def affiliations(self):
        afil_str = ""
        for afil in self.afs["AF"]:
            aid = afil.get("ID", None)
            name = afil["NAME"]
            afil_str += f"({aid}) {name}; "
        return afil_str
