import os
import requests
import time


class EngineeringVillageAPI:
    def __init__(self, apikey=None, insttoken=None):

        if apikey is None:
            apikey = os.environ.get("ELSAPI")
        if insttoken is None:
            insttoken = os.environ.get("ELSINST")
        self.apikey = apikey
        self.insttoken = insttoken
        self.base_url = "https://api.elsevier.com/content/ev"

    @property
    def headers(self) -> dict:
        """Helper property to build the headers object for the requests

        Returns
        -------
        dict
            header object for rest request
        """
        headers = {}
        if self.apikey:
            headers["X-ELS-APIKey"] = self.apikey
        if self.insttoken:
            headers["X-ELS-Insttoken"] = self.insttoken
        return headers

    def _get_request(self, url: str):
        """Sends a request to the Engineering village with headers
        It will try 5 times to query EV, if it fails will raise exception

        Parameters
        ----------
        url : url
            _description_

        Returns
        -------
        response
            response object

        Raises
        ------
        BaseException
            When search cannot be completed
        """
        attempts = 5
        resp = None
        while attempts > 0:
            try:
                resp = requests.get(url, headers=self.headers)
                resp.raise_for_status()  # if its not 200, raise an error
                attempts = 0
            except Exception as e:
                print(
                    f"Couldn't get a response, waiting 10 seconds and trying {attempts-1} more times"
                )
                print(e)
                # print(resp.text)
                # wait 10 seconds and try again
                time.sleep(10)
                attempts -= 1
        if resp:
            return resp
        else:
            raise BaseException(f"Cannot retrieve query: {url}")
