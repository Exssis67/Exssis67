from appJar import gui

_app = gui(handleArgs=False)
_counter = 0
_price_dict = dict()
_price = 0;

def init():
    _app.addLabel("price", "Preis: 0€")
    _app.setLabelBg("price", "green")
    setupPrices()
    _app.go()


def addLabel(content):
    global _counter, _price_dict, _price
    if not _app.alive:
        return
    if not content in _price_dict:
        return
    if content in _price_dict:
        _price += _price_dict[content]
        content = content + " - " + str(_price_dict[content]) + "€"
        _app.setLabel("price", "Preis: " + str(_price) + "€")
    _app.addLabel(str(_counter), content)
    _app.setLabelBg(str(_counter), "red")
    if _counter >= 10:
        _app.removeLabel(str(_counter - 10))
    _counter += 1


def setupPrices():
    setPrice("person", 10)
    setPrice("tvmonitor", 100)
    setPrice("knife", 30)
    setPrice("scissors", 50)
    setPrice("chair", 72)


def setPrice(product, price):
    global _price_dict
    o = {product:price}
    _price_dict.update(o)