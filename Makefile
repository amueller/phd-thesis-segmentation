all:
	rubber -c 'setlist arguments --shell-escape'  -I itm/images -I mil/images -I partial_supervision -I mil -I exact_learning -I pystruct --pdf main.tex
