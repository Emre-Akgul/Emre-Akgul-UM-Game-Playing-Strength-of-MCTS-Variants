import numpy as np
import pandas as pd


def extract_features(agent_column):
    """
    Getting the selection, exploration, playout and bounds from the agent columns
    Function to extract features based on the pattern provided
    """
    selection = agent_column.str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', expand=True)[0].astype('category')
    exploration = agent_column.str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', expand=True)[1].astype(float)
    playout = agent_column.str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', expand=True)[2].astype('category')
    bounds = agent_column.str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', expand=True)[3].astype('category')
    return selection, exploration, playout, bounds

def initial_preprocessing(df):
    """
    Apply initial preprocessing. 
    Remove features that are unnecessary in df and returns it.
    Details are on ExploratoryDataAnalysis/1-UnderstandingAssignmentAndFeatures.ipynb
    """

    # Applying the function to extract features for agent1 and agent2
    df['p1_selection'], df['p1_exploration'], df['p1_playout'], df['p1_bounds'] = extract_features(df['agent1'])
    df['p2_selection'], df['p2_exploration'], df['p2_playout'], df['p2_bounds'] = extract_features(df['agent2'])

    df = df.drop(["agent1", "agent2"], axis=1) 

    # Keep only columns that give new information.
    # Filter by the reduced columns obtained in the Exploratory Data Analysis - Understanding Assignment and Features
    df = df[["Stochastic", "AsymmetricPiecesType", "Team", "Shape", "SquareShape", "HexShape", "TriangleShape", "DiamondShape", "RectangleShape", "StarShape", "RegularShape", "PolygonShape", "Tiling", "SquareTiling", "HexTiling", "TriangleTiling", "SemiRegularTiling", "MorrisTiling", "CircleTiling", "ConcentricTiling", "SpiralTiling", "AlquerqueTiling", "MancalaStores", "MancalaTwoRows", "MancalaThreeRows", "MancalaFourRows", "MancalaSixRows", "MancalaCircular", "AlquerqueBoard", "AlquerqueBoardWithOneTriangle", "AlquerqueBoardWithTwoTriangles", "AlquerqueBoardWithFourTriangles", "AlquerqueBoardWithEightTriangles", "ThreeMensMorrisBoard", "ThreeMensMorrisBoardWithTwoTriangles", "NineMensMorrisBoard", "StarBoard", "CrossBoard", "KintsBoard", "PachisiBoard", "FortyStonesWithFourGapsBoard", "Track", "TrackLoop", "TrackOwned", "Region", "Boardless", "Vertex", "Cell", "Edge", "NumPlayableSitesOnBoard", "NumColumns", "NumRows", "NumCorners", "NumDirections", "NumOrthogonalDirections", "NumDiagonalDirections", "NumAdjacentDirections", "NumOffDiagonalDirections", "NumInnerSites", "NumLayers", "NumEdges", "NumCells", "NumVertices", "NumPerimeterSites", "NumTopSites", "NumBottomSites", "NumRightSites", "NumLeftSites", "NumCentreSites", "NumConvexCorners", "NumConcaveCorners", "NumPhasesBoard", "Hand", "NumContainers", "NumPlayableSites", "Piece", "PieceValue", "PieceDirection", "DiceD2", "DiceD4", "DiceD6", "LargePiece", "Tile", "NumComponentsType", "NumComponentsTypePerPlayer", "NumDice", "Meta", "SwapOption", "Repetition", "TurnKo", "PositionalSuperko", "Start", "PiecesPlacedOnBoard", "PiecesPlacedOutsideBoard", "InitialRandomPlacement", "InitialScore", "InitialCost", "NumStartComponentsBoard", "NumStartComponentsHand", "NumStartComponents", "Moves", "MovesDecision", "NoSiteMoves", "VoteDecision", "SwapPlayersDecision", "SwapPlayersDecisionFrequency", "PassDecision", "PassDecisionFrequency", "ProposeDecision", "ProposeDecisionFrequency", "SingleSiteMoves", "AddDecision", "AddDecisionFrequency", "PromotionDecision", "PromotionDecisionFrequency", "RemoveDecision", "RemoveDecisionFrequency", "RotationDecision", "TwoSitesMoves", "StepDecision", "StepDecisionFrequency", "StepDecisionToEmpty", "StepDecisionToEmptyFrequency", "StepDecisionToFriend", "StepDecisionToFriendFrequency", "StepDecisionToEnemy", "StepDecisionToEnemyFrequency", "SlideDecision", "SlideDecisionFrequency", "SlideDecisionToEmpty", "SlideDecisionToEmptyFrequency", "SlideDecisionToEnemy", "SlideDecisionToEnemyFrequency", "SlideDecisionToFriend", "SlideDecisionToFriendFrequency", "LeapDecision", "LeapDecisionFrequency", "LeapDecisionToEmpty", "LeapDecisionToEmptyFrequency", "LeapDecisionToEnemy", "LeapDecisionToEnemyFrequency", "HopDecision", "HopDecisionFrequency", "HopDecisionMoreThanOne", "HopDecisionMoreThanOneFrequency", "HopDecisionEnemyToEmpty", "HopDecisionEnemyToEmptyFrequency", "HopDecisionFriendToEmpty", "HopDecisionFriendToEmptyFrequency", "HopDecisionFriendToFriendFrequency", "HopDecisionEnemyToEnemy", "HopDecisionEnemyToEnemyFrequency", "HopDecisionFriendToEnemy", "HopDecisionFriendToEnemyFrequency", "FromToDecision", "FromToDecisionFrequency", "FromToDecisionWithinBoardFrequency", "FromToDecisionBetweenContainersFrequency", "FromToDecisionEmpty", "FromToDecisionEmptyFrequency", "FromToDecisionEnemy", "FromToDecisionEnemyFrequency", "FromToDecisionFriend", "FromToDecisionFriendFrequency", "SwapPiecesDecision", "SwapPiecesDecisionFrequency", "ShootDecision", "MovesNonDecision", "MovesEffects", "VoteEffect", "SwapPlayersEffect", "PassEffect", "Roll", "RollFrequency", "ProposeEffect", "ProposeEffectFrequency", "AddEffect", "AddEffectFrequency", "SowFrequency", "SowWithEffect", "SowCapture", "SowCaptureFrequency", "SowRemove", "SowRemoveFrequency", "SowBacktracking", "SowBacktrackingFrequency", "SowSkip", "SowOriginFirst", "SowCW", "SowCCW", "PromotionEffect", "PromotionEffectFrequency", "RemoveEffect", "RemoveEffectFrequency", "PushEffect", "PushEffectFrequency", "Flip", "FlipFrequency", "SetMove", "SetNextPlayer", "SetNextPlayerFrequency", "MoveAgain", "MoveAgainFrequency", "SetValue", "SetValueFrequency", "SetCount", "SetCountFrequency", "SetRotation", "StepEffect", "SlideEffect", "LeapEffect", "HopEffect", "FromToEffect", "MovesOperators", "Priority", "ByDieMove", "MaxMovesInTurn", "MaxDistance", "Capture", "ReplacementCapture", "ReplacementCaptureFrequency", "HopCapture", "HopCaptureFrequency", "HopCaptureMoreThanOne", "HopCaptureMoreThanOneFrequency", "DirectionCapture", "DirectionCaptureFrequency", "EncloseCapture", "EncloseCaptureFrequency", "CustodialCapture", "CustodialCaptureFrequency", "InterveneCapture", "InterveneCaptureFrequency", "SurroundCapture", "SurroundCaptureFrequency", "CaptureSequence", "CaptureSequenceFrequency", "Conditions", "SpaceConditions", "Line", "Connection", "Group", "Contains", "Pattern", "Fill", "Distance", "MoveConditions", "NoMoves", "NoMovesMover", "NoMovesNext", "CanMove", "CanNotMove", "PieceConditions", "NoPiece", "NoPieceMover", "NoPieceNext", "NoTargetPiece", "Threat", "IsEmpty", "IsEnemy", "IsFriend", "IsPieceAt", "LineOfSight", "CountPiecesComparison", "CountPiecesMoverComparison", "CountPiecesNextComparison", "ProgressCheck", "Directions", "AbsoluteDirections", "AllDirections", "AdjacentDirection", "OrthogonalDirection", "DiagonalDirection", "RotationalDirection", "SameLayerDirection", "RelativeDirections", "ForwardDirection", "BackwardDirection", "ForwardsDirection", "BackwardsDirection", "LeftwardDirection", "LeftwardsDirection", "ForwardRightDirection", "BackwardRightDirection", "SameDirection", "OppositeDirection", "Phase", "NumPlayPhase", "Scoring", "PieceCount", "SumDice", "SpaceEnd", "LineEnd", "LineEndFrequency", "LineWin", "LineWinFrequency", "LineLoss", "LineLossFrequency", "LineDraw", "ConnectionEnd", "ConnectionEndFrequency", "ConnectionWin", "ConnectionWinFrequency", "ConnectionLoss", "ConnectionLossFrequency", "GroupEnd", "GroupEndFrequency", "GroupWin", "GroupWinFrequency", "GroupLoss", "GroupDraw", "LoopEnd", "LoopWin", "LoopWinFrequency", "PatternWin", "PatternWinFrequency", "PathExtentLoss", "TerritoryWin", "TerritoryWinFrequency", "CaptureEnd", "Checkmate", "CheckmateFrequency", "CheckmateWin", "CheckmateWinFrequency", "NoTargetPieceEnd", "NoTargetPieceEndFrequency", "NoTargetPieceWin", "NoTargetPieceWinFrequency", "EliminatePiecesEnd", "EliminatePiecesEndFrequency", "EliminatePiecesWin", "EliminatePiecesWinFrequency", "EliminatePiecesLoss", "EliminatePiecesLossFrequency", "EliminatePiecesDraw", "EliminatePiecesDrawFrequency", "RaceEnd", "NoOwnPiecesEnd", "NoOwnPiecesEndFrequency", "NoOwnPiecesWin", "NoOwnPiecesWinFrequency", "NoOwnPiecesLoss", "NoOwnPiecesLossFrequency", "FillEnd", "FillEndFrequency", "FillWin", "FillWinFrequency", "ReachEnd", "ReachEndFrequency", "ReachWin", "ReachWinFrequency", "ReachLoss", "ReachLossFrequency", "ReachDraw", "ReachDrawFrequency", "ScoringEnd", "ScoringEndFrequency", "ScoringWin", "ScoringWinFrequency", "ScoringLoss", "ScoringLossFrequency", "ScoringDraw", "NoMovesEnd", "NoMovesEndFrequency", "NoMovesWin", "NoMovesWinFrequency", "NoMovesLoss", "NoMovesLossFrequency", "NoMovesDraw", "NoMovesDrawFrequency", "NoProgressEnd", "NoProgressDraw", "NoProgressDrawFrequency", "Draw", "DrawFrequency", "Misere", "DurationActions", "DurationMoves", "DurationTurns", "DurationTurnsStdDev", "DurationTurnsNotTimeouts", "DecisionMoves", "GameTreeComplexity", "StateTreeComplexity", "BoardCoverageDefault", "BoardCoverageFull", "BoardCoverageUsed", "AdvantageP1", "Balance", "Completion", "Drawishness", "Timeouts", "OutcomeUniformity", "BoardSitesOccupiedAverage", "BoardSitesOccupiedMedian", "BoardSitesOccupiedMaximum", "BoardSitesOccupiedVariance", "BoardSitesOccupiedChangeAverage", "BoardSitesOccupiedChangeSign", "BoardSitesOccupiedChangeLineBestFit", "BoardSitesOccupiedChangeNumTimes", "BoardSitesOccupiedMaxIncrease", "BoardSitesOccupiedMaxDecrease", "BranchingFactorAverage", "BranchingFactorMedian", "BranchingFactorMaximum", "BranchingFactorVariance", "BranchingFactorChangeAverage", "BranchingFactorChangeSign", "BranchingFactorChangeLineBestFit", "BranchingFactorChangeNumTimesn", "BranchingFactorChangeMaxIncrease", "BranchingFactorChangeMaxDecrease", "DecisionFactorAverage", "DecisionFactorMedian", "DecisionFactorMaximum", "DecisionFactorVariance", "DecisionFactorChangeAverage", "DecisionFactorChangeSign", "DecisionFactorChangeLineBestFit", "DecisionFactorChangeNumTimes", "DecisionFactorMaxIncrease", "DecisionFactorMaxDecrease", "MoveDistanceAverage", "MoveDistanceMedian", "MoveDistanceMaximum", "MoveDistanceVariance", "MoveDistanceChangeAverage", "MoveDistanceChangeSign", "MoveDistanceChangeLineBestFit", "MoveDistanceChangeNumTimes", "MoveDistanceMaxIncrease", "MoveDistanceMaxDecrease", "PieceNumberAverage", "PieceNumberMedian", "PieceNumberMaximum", "PieceNumberVariance", "PieceNumberChangeAverage", "PieceNumberChangeSign", "PieceNumberChangeLineBestFit", "PieceNumberChangeNumTimes", "PieceNumberMaxIncrease", "PieceNumberMaxDecrease", "ScoreDifferenceAverage", "ScoreDifferenceMedian", "ScoreDifferenceMaximum", "ScoreDifferenceVariance", "ScoreDifferenceChangeAverage", "ScoreDifferenceChangeSign", "ScoreDifferenceChangeLineBestFit", "ScoreDifferenceMaxIncrease", "ScoreDifferenceMaxDecrease", "Math", "Arithmetic", "Operations", "Addition", "Subtraction", "Multiplication", "Division", "Modulo", "Absolute", "Exponentiation", "Minimum", "Maximum", "Comparison", "Equal", "NotEqual", "LesserThan", "LesserThanOrEqual", "GreaterThan", "GreaterThanOrEqual", "Parity", "Even", "Odd", "Logic", "Conjunction", "Disjunction", "Negation", "Set", "Union", "Intersection", "Complement", "Algorithmics", "ConditionalStatement", "ControlFlowStatement", "Visual", "Style", "BoardStyle", "GraphStyle", "ChessStyle", "GoStyle", "MancalaStyle", "PenAndPaperStyle", "ShibumiStyle", "BackgammonStyle", "JanggiStyle", "XiangqiStyle", "ShogiStyle", "TableStyle", "SurakartaStyle", "TaflStyle", "NoBoard", "ComponentStyle", "AnimalComponent", "ChessComponent", "KingComponent", "QueenComponent", "KnightComponent", "RookComponent", "BishopComponent", "PawnComponent", "FairyChessComponent", "PloyComponent", "ShogiComponent", "XiangqiComponent", "StrategoComponent", "JanggiComponent", "CheckersComponent", "BallComponent", "TaflComponent", "DiscComponent", "MarkerComponent", "StackType", "Stack", "Symbols", "ShowPieceValue", "ShowPieceState", "Implementation", "State", "StackState", "PieceState", "SiteState", "SetSiteState", "VisitedSites", "Variable", "SetVar", "RememberValues", "ForgetValues", "SetPending", "InternalCounter", "SetInternalCounter", "PlayerValue", "Efficiency", "CopyContext", "Then", "ForEachPiece", "DoLudeme", "Trigger", "PlayoutsPerSecond", "MovesPerSecond", "num_wins_agent1", "num_draws_agent1", "num_losses_agent1", "utility_agent1", "p1_selection", "p1_exploration", "p1_playout", "p1_bounds", "p2_selection", "p2_exploration", "p2_playout", "p2_bounds"]]
    
    return df

def feature_elimination(df):
    features = ['Balance',
                'AdvantageP1',
                'Completion',
                'OutcomeUniformity',
                'NumOrthogonalDirections',
                'MovesPerSecond',
                'PlayoutsPerSecond',
                'DurationTurnsStdDev',
                'PieceNumberChangeAverage',
                'MoveDistanceMedian',
                'NumAdjacentDirections',
                'NumCells',
                'PieceNumberAverage',
                'BranchingFactorMedian',
                'DurationTurnsNotTimeouts',
                'DecisionFactorMaximum',
                'NumColumns',
                'StateTreeComplexity',
                'MoveDistanceMaximum',
                'BoardCoverageFull',
                'NumPerimeterSites',
                'NumDiagonalDirections',
                'NumRows',
                'RemoveEffectFrequency',
                'PieceNumberVariance',
                'NumStartComponents',
                "num_wins_agent1",
                "num_draws_agent1",
                "num_losses_agent1",
                "utility_agent1",
                ]
    
    df = df[features]

    return df


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the X and y into training and testing.
    """

    # Shuffle the data
    shuffled_indices = X.sample(frac=1, random_state=random_state).index
    X_shuffled = X.loc[shuffled_indices]
    y_shuffled = y.loc[shuffled_indices]
    
    # Calculate 
    test_size_count = int(test_size * len(X))
    
    # Split data
    X_train = X_shuffled.iloc[:-test_size_count]
    X_test = X_shuffled.iloc[-test_size_count:]
    y_train = y_shuffled.iloc[:-test_size_count]
    y_test = y_shuffled.iloc[-test_size_count:]
    
    return X_train, X_test, y_train, y_test

def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """
    Calculate root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

