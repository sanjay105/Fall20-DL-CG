import numpy as np
def printboard(board):
    tmp=[[]]
    for i in range(9):
        if board[0][i]==0:
            tmp[0].append(' ')
        elif board[0][i]==-1:
            tmp[0].append('O')
        else:
            tmp[0].append('X')
    board=tmp
    print('Your Tic-Tac-Toe board now:\nO -> Computer Move X -> User Move')
    print("-------------")
    print("| "+board[0][0] + " | " + board[0][1] + " | " + board[0][2]+" |")
    print("-------------")
    print("| "+board[0][3] + " | " + board[0][4] + " | " + board[0][5]+" |")
    print("-------------")
    print("| "+board[0][6] + " | " + board[0][7] + " | " + board[0][8]+" |")
    print("-------------")

def CheckVictory(board):
    if board[0][0]==board[0][1]==board[0][2]!=0:
        return True
    if board[0][0]==board[0][3]==board[0][6]!=0:
        return True
    if board[0][0]==board[0][4]==board[0][8]!=0:
        return True
    if board[0][4]==board[0][1]==board[0][7]!=0:
        return True
    if board[0][8]==board[0][5]==board[0][2]!=0:
        return True
    if board[0][6]==board[0][4]==board[0][2]!=0:
        return True
    if board[0][3]==board[0][4]==board[0][5]!=0:
        return True
    if board[0][8]==board[0][7]==board[0][6]!=0:
        return True
    return False


def adjustRegressorValues(y_pred, ftype = None):
    '''

    :param y_pred: predicted y values for X_test
    :param ftype: file type that we are working upon - {'single','multi','final'}
    :return: returns adjust regressor values
    '''
    if ftype == 'final':
        y_pred[y_pred<0] = -1
        y_pred[y_pred>=0] = 1
    elif ftype == 'single':
        y_pred[y_pred<=0] = 0
        y_pred[y_pred>=8] = 8
        y_pred = np.floor(y_pred+.5)
    else:
        y_val = np.zeros((y_pred.shape[0], 1))
        for i,vals in enumerate(y_pred):
            mx, idx = -10000000, -1
            for j in range(len(vals)):
                if mx<vals[j]:
                    mx = vals[j]
                    idx = j
            y_val[i] = idx
        y_pred = y_val
    return y_pred

def gameplay(play_model,board,isCompMove, isRegressor = False, ftype = None):
    isWinner = False
    for i in range(9):
        if isCompMove:
            move=play_model.predict(board)
            if isRegressor == True:
                move = adjustRegressorValues(move, ftype)
            printboard(board)
            board[0][int(move[0])]=-1
            if CheckVictory(board):
                isWinner=True
                printboard(board)
                print("Computer Won")
                break
        else:
            printboard(board)
            move = int(input("Enter the index to insert:"))
            while(board[0][move]!=0):
                move = int(input("Invalid Move!!\nEnter the index to insert:"))
            board[0][move]=1
            if CheckVictory(board):
                isWinner=True
                printboard(board)
                print("You Won")
                break
        isCompMove = not isCompMove
    if not isWinner:
        print("Draw")



def game(play_model):
    board = [[0,0,0,0,0,0,0,0,0]]
    inp= int(input("Enter \n1. To start Computer\n2. To start your move"))
    gameplay(play_model,board,inp==1)