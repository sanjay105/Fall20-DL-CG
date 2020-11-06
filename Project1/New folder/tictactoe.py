import pickle
from gameplay import game

def main():
    model=pickle.load(open('gamemodel.mdl','rb'))
    game(model)

if __name__ == "__main__":
    main()