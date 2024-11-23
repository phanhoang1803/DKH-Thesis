# import socket
# import copy
# import threading
# import logging

# import requests
# import urllib3

# from .response_decoder import decode_response

# MAX_FILE_SIZE = 20000000
# MIN_FILE_SIZE = 10

# LOGGER = logging.getLogger(__name__)

# # user agent
# USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36"

# # customize headers
# HEADERS = {
#     "Connection": "close",
#     "User-Agent": USER_AGENT,
# }
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# class SimpleCrawler(object):
#     _results = {}

#     @staticmethod
#     def fetch_url(url, timeout=None, user_agent=USER_AGENT):
#         """
#         Crawls the html content of the parameter url and returns the html
#         :param url:
#         :param timeout: in seconds, if None, the urllib default is used
#         :return:
#         """
#         return SimpleCrawler._fetch_url(url, False, timeout=timeout, user_agent=user_agent)

#     @staticmethod
#     def _fetch_url(url, is_threaded, timeout=None, user_agent=USER_AGENT):
#         """
#         Crawls the html content of the parameter url and saves the html in _results
#         :param url:
#         :param is_threaded: If True, results will be stored for later processing by the fetch_urls method. Else not.
#         :param timeout: in seconds, if None, the urllib default is used
#         :return: html of the url
#         """
#         headers = HEADERS
#         if user_agent:
#             headers["User-Agent"] = user_agent

#         html_str = None
#         # send
#         try:
#             # read by streaming chunks (stream=True, iter_content=xx)
#             # so we can stop downloading as soon as MAX_FILE_SIZE is reached
#             response = requests.get(
#                 url,
#                 timeout=timeout,
#                 verify=False,
#                 allow_redirects=True,
#                 headers=headers,
#             )
#         except (requests.exceptions.MissingSchema, requests.exceptions.InvalidURL):
#             LOGGER.error("malformed URL: %s", url)
#         except requests.exceptions.TooManyRedirects:
#             LOGGER.error("too many redirects: %s", url)
#         except requests.exceptions.SSLError as err:
#             LOGGER.error("SSL: %s %s", url, err)
#         except (
#             socket.timeout,
#             requests.exceptions.ConnectionError,
#             requests.exceptions.Timeout,
#             socket.error,
#             socket.gaierror,
#         ) as err:
#             LOGGER.error("connection/timeout error: %s %s", url, err)
#         else:
#             # safety checks
#             if response.status_code != 200:
#                 LOGGER.error("not a 200 response: %s", response.status_code)
#             elif response.text is None or len(response.text) < MIN_FILE_SIZE:
#                 LOGGER.error("too small/incorrect: %s %s", url, len(response.text))
#             elif len(response.text) > MAX_FILE_SIZE:
#                 LOGGER.error("too large: %s %s", url, len(response.text))
#             else:
#                 html_str = decode_response(response)
#         if is_threaded:
#             SimpleCrawler._results[url] = html_str
#         return html_str

#     @staticmethod
#     def fetch_urls(urls, timeout=None, user_agent=USER_AGENT):
#         """
#         Crawls the html content of all given urls in parallel. Returns when all requests are processed.
#         :param urls:
#         :param timeout: in seconds, if None, the urllib default is used
#         :return:
#         """
#         threads = [
#             threading.Thread(target=SimpleCrawler._fetch_url, args=(url, True, timeout, user_agent))
#             for url in urls
#         ]
#         for thread in threads:
#             thread.start()
#         for thread in threads:
#             thread.join()

#         results = copy.deepcopy(SimpleCrawler._results)
#         SimpleCrawler._results = {}
#         return results


import socket
import copy
import threading
import logging
from seleniumbase import Driver

from .response_decoder import decode_response

MAX_FILE_SIZE = 20000000
MIN_FILE_SIZE = 10

LOGGER = logging.getLogger(__name__)

# user agent
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36"


class SimpleCrawler(object):
    _results = {}

    @staticmethod
    def fetch_url(url, timeout=None, user_agent=USER_AGENT):
        """
        Crawls the html content of the parameter url and returns the html
        :param url:
        :param timeout: in seconds, if None, the default is used
        :return:
        """
        return SimpleCrawler._fetch_url(url, False, timeout=timeout, user_agent=user_agent)

    @staticmethod
    def _fetch_url(url, is_threaded, timeout=None, user_agent=USER_AGENT):
        """
        Crawls the html content of the parameter url and saves the html in _results
        :param url:
        :param is_threaded: If True, results will be stored for later processing by the fetch_urls method. Else not.
        :param timeout: in seconds, if None, the default is used
        :return: html of the url
        """
        html_str = None
        # Fetch using SeleniumBase UC mode
        try:
            options = {
                "uc": True,  # Enable undetected Chrome mode
                "headless": True,  # Run in headless mode
                "incognito": True,
            }
            with Driver(**options) as driver:
                driver.get(url)
                html_str = driver.get_page_source()

                # Safety checks
                if html_str is None or len(html_str) < MIN_FILE_SIZE:
                    LOGGER.error("too small/incorrect: %s %s", url, len(html_str or ""))
                    html_str = None
                elif len(html_str) > MAX_FILE_SIZE:
                    LOGGER.error("too large: %s %s", url, len(html_str))
                    html_str = None

        except Exception as err:
            LOGGER.error("Error fetching URL with Selenium: %s %s", url, err)

        if is_threaded:
            SimpleCrawler._results[url] = html_str
        return html_str

    @staticmethod
    def fetch_urls(urls, timeout=None, user_agent=USER_AGENT):
        """
        Crawls the html content of all given urls in parallel. Returns when all requests are processed.
        :param urls:
        :param timeout: in seconds, if None, the default is used
        :return:
        """
        threads = [
            threading.Thread(target=SimpleCrawler._fetch_url, args=(url, True, timeout, user_agent))
            for url in urls
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        results = copy.deepcopy(SimpleCrawler._results)
        SimpleCrawler._results = {}
        return results