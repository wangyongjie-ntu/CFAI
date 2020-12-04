Tutorials
=====

Python 代码
------------------
这里是 Python 代码::

	# coding=utf-8
	import requests

	s = requests.Session()

	html = s.get("http://www.baidu.com")

	print html.content


C 代码
-----------------------
这里是 C 代码

.. code-block:: C

	#include "stdio.h"

	int main(void){
		printf("Hello world");
		return 0;
	}

在双冒号之后要空行
