from practica7 import TopK

if __name__=='__main__':
    a=TopK('ex7data.txt',20)
    solucion=['[]']
    solucion.append('[\'Female\']')
    solucion.append('[\'DoNotOwnHome\']')
    solucion.append('[\'Homeowner\']')
    solucion.append('[\'Male\']')
    solucion.append('[\'cannedveg\']')
    solucion.append('[\'frozenmeal\']')
    solucion.append('[\'fruitveg\']')
    solucion.append('[\'beer\']')
    solucion.append('[\'fish\']')
    solucion.append('[\'wine\']')
    solucion.append('[\'confectionery\']')
    solucion.append('[\'Female\', \'Homeowner\']')
    solucion.append('[\'DoNotOwnHome\', \'Female\']')
    solucion.append('[\'DoNotOwnHome\', \'Male\']')
    solucion.append('[\'Homeowner\', \'Male\']')
    solucion.append('[\'Male\', \'frozenmeal\']')
    solucion.append('[\'cannedmeat\']')
    solucion.append('[\'Male\', \'cannedveg\']')
    solucion.append('[\'DoNotOwnHome\', \'fish\']')
    solucion.append('[\'Male\', \'beer\']')
    for num,e in enumerate(a):
        print("Solucion:",solucion[num]," Valor del programa:",sorted(e))
