import numpy as np
import copy


def feat_v1(obs1):
    obs = copy.deepcopy(obs1)
    maps = []
    board = obs['board']

    '''爆炸one-hot'''
    bomb_life = obs['bomb_life']
    bomb_blast_strength = obs['bomb_blast_strength']
    flame_life = obs['flame_life']
    # 统一炸弹时间
    for x in range(11):
        for y in range(11):
            if bomb_blast_strength[(x, y)] > 0:
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x - i, y)
                    if x - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y + i)
                    if y + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y - i)
                    if y - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
    bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
    flame_life = np.where(flame_life == 0, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)
    for i in range(2, 13):
        maps.append(bomb_life == i)

    '''将bomb direction编码为one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction']
    for i in range(1, 5):
        maps.append(bomb_moving_direction == i)

    """棋盘物体 one hot"""
    for i in range(9):  # [0, 1, ..., 8]
        maps.append(board == i)
    """标量映射为11*11的矩阵"""
    maps.append(np.full(board.shape, obs['ammo']) / 5)
    maps.append(np.full(board.shape, obs['blast_strength']) / 5)
    maps.append(np.full(board.shape, obs['can_kick']))

    # 队友
    teammate_idx = obs['teammate'].value
    maps.append(board == teammate_idx)
    # 我
    my_idx = (teammate_idx - 10 + 2) % 4 + 10
    maps.append(board == my_idx)
    # 俩敌人
    enemy_1_idx = (teammate_idx - 10 + 1) % 4 + 10
    enemy_2_idx = (teammate_idx - 10 + 3) % 4 + 10
    maps.append(np.logical_or(board == enemy_1_idx, board == enemy_2_idx))

    return np.stack(maps)


def get_feature_shape_v1():
    return 30, 11, 11


def feat_v2(obs1):
    obs = copy.deepcopy(obs1)
    maps = []
    board = obs['board']

    '''爆炸one-hot'''
    bomb_life = obs['bomb_life']
    bomb_blast_strength = obs['bomb_blast_strength']
    flame_life = obs['flame_life']
    # 统一炸弹时间
    for x in range(11):
        for y in range(11):
            if bomb_blast_strength[(x, y)] > 0:
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x - i, y)
                    if x - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y + i)
                    if y + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y - i)
                    if y - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
    bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
    flame_life = np.where(flame_life == 0, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)
    for i in range(2, 13):
        maps.append(bomb_life == i)

    '''将bomb direction编码为one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction']
    for i in range(1, 5):
        maps.append(bomb_moving_direction == i)

    """棋盘物体 one hot"""
    for i in range(9):  # [0, 1, ..., 8]
        maps.append(board == i)

    # 队友
    teammate_idx = obs['teammate'].value
    maps.append(board == teammate_idx)
    # 我
    my_idx = (teammate_idx - 10 + 2) % 4 + 10
    maps.append(board == my_idx)
    # 俩敌人
    enemy_1_idx = (teammate_idx - 10 + 1) % 4 + 10
    enemy_2_idx = (teammate_idx - 10 + 3) % 4 + 10
    maps.append(np.logical_or(board == enemy_1_idx, board == enemy_2_idx))

    """标量映射为11*11的矩阵"""
    # maps.append(np.full(board.shape, obs['ammo']) / 5)
    # maps.append(np.full(board.shape, obs['blast_strength']) / 5)
    # maps.append(np.full(board.shape, obs['can_kick']))
    ammo = obs['ammo'] / 5
    blast_strength = obs['blast_strength'] / 5
    can_kick = obs['can_kick']
    # teammate_alive =
    scalar = np.array([ammo, blast_strength, can_kick])

    return [np.stack(maps), scalar]


def get_feature_shape_v2():
    return 27, 11, 11


def feat_v3(obs1):
    obs = copy.deepcopy(obs1)
    maps = []
    board = obs['board']

    '''爆炸one-hot'''
    bomb_life = obs['bomb_life']
    bomb_blast_strength = obs['bomb_blast_strength']
    flame_life = obs['flame_life']
    # 统一炸弹时间
    for x in range(11):
        for y in range(11):
            if bomb_blast_strength[(x, y)] > 0:
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x - i, y)
                    if x - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y + i)
                    if y + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y - i)
                    if y - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
    bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
    flame_life = np.where(flame_life == 0, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)
    for i in range(2, 13):
        maps.append(bomb_life == i)

    '''将bomb direction编码为one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction']
    for i in range(1, 5):
        maps.append(bomb_moving_direction == i)

    """棋盘物体 one hot"""
    for i in range(9):  # [0, 1, ..., 8]
        maps.append(board == i)

    # 队友
    teammate_idx = obs['teammate'].value
    maps.append(board == teammate_idx)
    # 我
    my_idx = (teammate_idx - 10 + 2) % 4 + 10
    maps.append(board == my_idx)
    # 俩敌人
    enemy_1_idx = (teammate_idx - 10 + 1) % 4 + 10
    enemy_2_idx = (teammate_idx - 10 + 3) % 4 + 10
    maps.append(np.logical_or(board == enemy_1_idx, board == enemy_2_idx))

    """标量映射为11*11的矩阵"""
    ammo = [0, 0, 0, 0, 0]
    idx = obs['ammo'] if obs['ammo'] <= 4 else 4
    ammo[idx] = 1

    blast_strength = [0, 0, 0, 0, 0, 0, 0]
    idx = obs['blast_strength']-1 if obs['blast_strength'] <= 7 else 7
    blast_strength[idx] = 1

    can_kick = [obs['can_kick']]

    teammate_alive = [teammate_idx in obs['alive']]

    scalar = np.array([ammo, blast_strength, can_kick, teammate_alive])

    return [np.stack(maps), scalar]


def get_feature_shape_v3():
    return 27, 11, 11


def featurize_bak(obs):
    maps = []
    board = obs['board']
    # maps.append(board)
    maps.append(board == 1)
    maps.append(board == 2)
    # maps.append(board == 3)
    maps.append(board == 4)
    maps.append(np.logical_or(board == 6, board == 7))
    maps.append(board == 8)
    # buff
    maps.append(np.full(board.shape, obs['ammo'])/5)
    maps.append(np.full(board.shape, obs['blast_strength'])/5)
    maps.append(np.full(board.shape, obs['can_kick']))
    # 队友
    teammate_idx = obs['teammate'].value
    maps.append(board == teammate_idx)
    # 我
    my_idx = (teammate_idx-10+2) % 4 + 10
    maps.append(board == my_idx)
    # 俩敌人
    enemy_1_idx = (teammate_idx-10+1) % 4 + 10
    enemy_2_idx = (teammate_idx-10+3) % 4 + 10
    maps.append(np.logical_or(board == enemy_1_idx, board == enemy_2_idx))

    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])
    maps.append(obs['flame_life'])
    return np.stack(maps)


def get_feature_shape_bak():
    return 14, 11, 11
