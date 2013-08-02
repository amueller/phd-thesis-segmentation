paper:
	mkdir -p .mk
	rubber -f -I . -I images --pdf --into .mk fullpaper
	ln -sf .mk/fullpaper.pdf .
talk:
	mkdir -p .mk
	rubber -f -I . -I images --pdf --into .mk talk
	ln -sf .mk/talk.pdf .

figures:
	rm .mk/paper-figure*.pdf
	make paper

all: fullpaper 

clean:
	rm -rf .mk
	rm -f paper.pdf
	rm -f talk.pdf
