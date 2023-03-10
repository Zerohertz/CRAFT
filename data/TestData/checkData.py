import os

def checkData():
    os.chdir('images')
    img = os.listdir()
    os.chdir('../json')
    json = os.listdir()

    tmpi = []

    for i in img:
        tmpi.append(i[:-4])

    tmpj = []

    for i in json:
        tmpj.append(i[:-5])

    for i in tmpi:
        if not i in tmpj and not 'ipynb' in i:
            print('NG: ', i)

    for i in tmpj:
        if not i in tmpi and not 'ipynb' in i:
            print('NG: ', i)

    print('Images: ', len(img))
    print('Ground Truth: ', len(json))
    
if __name__ == "__main__":
    print(os.listdir())
    tmp = input()
    os.chdir(tmp)
    checkData()