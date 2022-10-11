from django.shortcuts import render
import pickle

def home(request):
    return render(request, 'index.html')

def getPredictions( GP, MIN, PTS, FGM, FGA, FGpercent, threePMade, threePA, threePpercent,FTM ,FTA,FTpercent,OREB, DREB,REB,AST,STL ,BLK,TOV):
    model = pickle.load(open('ml_model.sav', 'rb'))
    scaled = pickle.load(open('scaler.sav', 'rb'))

    prediction = model.predict(scaled.transform([
        [ GP, MIN, PTS, FGM, FGA, FGpercent, threePMade, threePA, threePpercent,FTM ,FTA,FTpercent,OREB, DREB,REB,AST,STL ,BLK,TOV ]
    ]))
    
    if (prediction == 0.):
        return 'no'
    elif (prediction == 1.):
        return 'yes'
    else:
        return 'error'

def result(request):
    GP = float(request.GET['GP'])
    MIN = float(request.GET['MIN'])
    PTS = float(request.GET['PTS'])
    FGM = float(request.GET['FGM'])
    FGA = float(request.GET['FGA'])
    FGpercent = float(request.GET['FGpercent'])
    threePMade = float(request.GET['threePMade'])
    threePA = float(request.GET['threePA'])
    threePpercent = float(request.GET['threePpercent'])
    FTM = float(request.GET['FTM'])
    FTA = float(request.GET['FTA'])
    FTpercent = float(request.GET['FTpercent'])
    OREB = float(request.GET['OREB'])
    DREB = float(request.GET['DREB'])
    REB = float(request.GET['REB'])
    AST = float(request.GET['AST'])
    STL = float(request.GET['STL'])
    BLK = float(request.GET['BLK'])
    TOV = float(request.GET['TOV'])


    result = getPredictions( GP, MIN, PTS, FGM, FGA, FGpercent, threePMade, threePA, threePpercent,FTM ,FTA,FTpercent,OREB, DREB,REB,AST,STL ,BLK,TOV)

    return render(request, 'result.html', {'result': result})
