Tutorials
=====

Python 
------------------

	# coding=utf-8
	import requests

	s = requests.Session()

	html = s.get("http://www.baidu.com")

	print html.content


