import math
import sys

# CoordinateTransformation v1.5
# import CoordinateTransformation as CoorTrans
# by wangjiaze on 2019-Apr-16
# v1.1 add function : Cartesian2Fractional, CalculateDistance, CalculateVolume, Groupnum2Groupname, Groupname2Groupnum
# v1.2 SymmetricOperation(coordinate, matrices) -> SymmetricOperation(coordinate, matrices, retain = 4)
# v1.3 add function :CalculateDistance2
# v1.4 add function : Groupnum2CrystalSystem, GetSymmetryEquivPosAsXyz
# v1.4.1 GetMatrix(SpaceGroupName, WycFilePath = sys.argv[0]+'/'+'wyckfull.dat')
# v1.5 add function : GetGeneralMatrix(spaceGroupName, WycFilePath = '/'.join(sys.argv[0].replace('\\','/').split('/')[:-1]) +'/'+'wyckfull.dat'):
#      add function : UniqueAtom2AllAtom(coordinates, spaceGroupName, retain = 6)

#  z = round(z0 * c * math.sqrt((1 - cosA * cosA - cosB * cosB - cosC * cosC + 2 * cosA * cosB * cosB)) / sinC, retain)
#  z = round(z0 * c * math.sqrt((1 - cosA * cosA - cosB * cosB - cosC * cosC + 2 * cosA * cosB * cosC)) / sinC, retain)


def Fractional2Cartesian(coordinate, CellParameter, retain = 6):
    x0 = coordinate[0]
    y0 = coordinate[1]
    z0 = coordinate[2]

    a = CellParameter[0]
    b = CellParameter[1]
    c = CellParameter[2]
    A = CellParameter[3]/180*math.pi
    B = CellParameter[4]/180*math.pi
    C = CellParameter[5]/180*math.pi

    cosA = math.cos(A)
    cosB = math.cos(B)
    sinC = math.sin(C)
    cosC = math.cos(C)

    x = round(a * x0 + b * cosC * y0 + c * cosB * z0, retain)
    y = round(y0 * b * sinC + c * (cosA - cosB * cosC) * z0 / sinC, retain)
    z = round(z0 * c * math.sqrt((1 - cosA * cosA - cosB * cosB - cosC * cosC + 2 * cosA * cosB * cosC)) / sinC, retain)

    return [x, y, z]


def Cartesian2Fractional(coordinate, CellParameter, retain = 6):
    x0 = coordinate[0]
    y0 = coordinate[1]
    z0 = coordinate[2]

    a = CellParameter[0]
    b = CellParameter[1]
    c = CellParameter[2]
    A = CellParameter[3]/180*math.pi
    B = CellParameter[4]/180*math.pi
    C = CellParameter[5]/180*math.pi

    cosA = math.cos(A)
    cosB = math.cos(B)
    sinC = math.sin(C)
    cosC = math.cos(C)

    x = round(x0 / a - b * cosC * y0 / a - c * cosB * z0 / a, retain)
    y = round(y0 / (b * sinC) - c * (cosA - cosB * cosC) * z0 / (b * sinC ** 2), retain)
    z = round(z0 * sinC / (c *math.sqrt(1-cosA**2 - cosB**2 -cosC**2 + 2*cosA * cosB * cosC)), retain)

    return [x, y, z]


def Groupnum2Groupname(Groupnum):
    Groupnum_Groupname = {1: 'P1', 2: 'P-1', 3: 'P2', 4: 'P21', 5: 'C2', 6: 'PM', 7: 'PC', 8: 'CM', 9: 'CC', 10: 'P2/M',
                          11: 'P21/M', 12: 'C2/M', 13: 'P2/C', 14: 'P21/C', 15: 'C2/C', 16: 'P222', 17: 'P2221',
                          18: 'P21212', 19: 'P212121', 20: 'C2221', 21: 'C222', 22: 'F222', 23: 'I222', 24: 'I212121',
                          25: 'PMM2', 26: 'PMC21', 27: 'PCC2', 28: 'PMA2', 29: 'PCA21', 30: 'PNC2', 31: 'PMN21',
                          32: 'PBA2', 33: 'PNA21', 34: 'PNN2', 35: 'CMM2', 36: 'CMC21', 37: 'CCC2', 38: 'AMM2',
                          39: 'AEM2', 40: 'AMA2', 41: 'AEA2', 42: 'FMM2', 43: 'FDD2', 44: 'IMM2', 45: 'IBA2',
                          46: 'IMA2', 47: 'PMMM', 48: 'PNNN', 49: 'PCCM', 50: 'PBAN', 51: 'PMMA', 52: 'PNNA',
                          53: 'PMNA', 54: 'PCCA', 55: 'PBAM', 56: 'PCCN', 57: 'PBCM', 58: 'PNNM', 59: 'PMMN',
                          60: 'PBCN', 61: 'PBCA', 62: 'PNMA', 63: 'CMCM', 64: 'CMCA', 65: 'CMMM', 66: 'CCCM',
                          67: 'CMMA', 68: 'CCCA', 69: 'FMMM', 70: 'FDDD', 71: 'IMMM', 72: 'IBAM', 73: 'IBCA',
                          74: 'IMMA', 75: 'P4', 76: 'P41', 77: 'P42', 78: 'P43', 79: 'I4', 80: 'I41', 81: 'P-4',
                          82: 'I-4', 83: 'P4/M', 84: 'P42/M', 85: 'P4/N', 86: 'P42/N', 87: 'I4/M', 88: 'I41/A',
                          89: 'P422', 90: 'P4212', 91: 'P4122', 92: 'P41212', 93: 'P4222', 94: 'P42212', 95: 'P4322',
                          96: 'P43212', 97: 'I422', 98: 'I4122', 99: 'P4MM', 100: 'P4BM', 101: 'P42CM', 102: 'P42NM',
                          103: 'P4CC', 104: 'P4NC', 105: 'P42MC', 106: 'P42BC', 107: 'I4MM', 108: 'I4CM', 109: 'I41MD',
                          110: 'I41CD', 111: 'P-42M', 112: 'P-42C', 113: 'P-421M', 114: 'P-421C', 115: 'P-4M2',
                          116: 'P-4C2', 117: 'P-4B2', 118: 'P-4N2', 119: 'I-4M2', 120: 'I-4C2', 121: 'I-42M',
                          122: 'I-42D', 123: 'P4/MMM', 124: 'P4/MCC', 125: 'P4/NBM', 126: 'P4/NNC', 127: 'P4/MBM',
                          128: 'P4/MNC', 129: 'P4/NMM', 130: 'P4/NCC', 131: 'P42/MMC', 132: 'P42/MCM', 133: 'P42/NBC',
                          134: 'P42/NNM', 135: 'P42/MBC', 136: 'P42/MNM', 137: 'P42/NMC', 138: 'P42/NCM', 139: 'I4/MMM',
                          140: 'I4/MCM', 141: 'I41/AMD', 142: 'I41/ACD', 143: 'P3', 144: 'P31', 145: 'P32', 146: 'R3',
                          147: 'P-3', 148: 'R-3', 149: 'P312', 150: 'P321', 151: 'P3112', 152: 'P3121', 153: 'P3212',
                          154: 'P3221', 155: 'R32', 156: 'P3M1', 157: 'P31M', 158: 'P3C1', 159: 'P31C', 160: 'R3M',
                          161: 'R3C', 162: 'P-31M', 163: 'P-31C', 164: 'P-3M1', 165: 'P-3C1', 166: 'R-3M', 167: 'R-3C',
                          168: 'P6', 169: 'P61', 170: 'P65', 171: 'P62', 172: 'P64', 173: 'P63', 174: 'P-6',
                          175: 'P6/M', 176: 'P63/M', 177: 'P622', 178: 'P6122', 179: 'P6522', 180: 'P6222',
                          181: 'P6422', 182: 'P6322', 183: 'P6MM', 184: 'P6CC', 185: 'P63CM', 186: 'P63MC',
                          187: 'P-6M2', 188: 'P-6C2', 189: 'P-62M', 190: 'P-62C', 191: 'P6/MMM', 192: 'P6/MCC',
                          193: 'P63/MCM', 194: 'P63/MMC', 195: 'P23', 196: 'F23', 197: 'I23', 198: 'P213', 199: 'I213',
                          200: 'PM-3', 201: 'PN-3', 202: 'FM-3', 203: 'FD-3', 204: 'IM-3', 205: 'PA-3', 206: 'IA-3',
                          207: 'P432', 208: 'P4232', 209: 'F432', 210: 'F4132', 211: 'I432', 212: 'P4332', 213: 'P4132',
                          214: 'I4132', 215: 'P-43M', 216: 'F-43M', 217: 'I-43M', 218: 'P-43N', 219: 'F-43C',
                          220: 'I-43D', 221: 'PM-3M', 222: 'PN-3N', 223: 'PM-3N', 224: 'PN-3M', 225: 'FM-3M',
                          226: 'FM-3C', 227: 'FD-3M', 228: 'FD-3C', 229: 'IM-3M', 230: 'IA-3D'}
    Groupname = Groupnum_Groupname[Groupnum]
    return Groupname


def Groupname2Groupnum(Groupname):
    Groupname_Groupnum = {'P1': 1, 'P-1': 2, 'P2': 3, 'P21': 4, 'C2': 5, 'PM': 6, 'PC': 7, 'CM': 8, 'CC': 9, 'P2/M': 10,
                          'P21/M': 11, 'C2/M': 12, 'P2/C': 13, 'P21/C': 14, 'C2/C': 15, 'P222': 16, 'P2221': 17,
                          'P21212': 18, 'P212121': 19, 'C2221': 20, 'C222': 21, 'F222': 22, 'I222': 23, 'I212121': 24,
                          'PMM2': 25, 'PMC21': 26, 'PCC2': 27, 'PMA2': 28, 'PCA21': 29, 'PNC2': 30, 'PMN21': 31,
                          'PBA2': 32, 'PNA21': 33, 'PNN2': 34, 'CMM2': 35, 'CMC21': 36, 'CCC2': 37, 'AMM2': 38,
                          'AEM2': 39, 'AMA2': 40, 'AEA2': 41, 'FMM2': 42, 'FDD2': 43, 'IMM2': 44, 'IBA2': 45,
                          'IMA2': 46, 'PMMM': 47, 'PNNN': 48, 'PCCM': 49, 'PBAN': 50, 'PMMA': 51, 'PNNA': 52,
                          'PMNA': 53, 'PCCA': 54, 'PBAM': 55, 'PCCN': 56, 'PBCM': 57, 'PNNM': 58, 'PMMN': 59,
                          'PBCN': 60, 'PBCA': 61, 'PNMA': 62, 'CMCM': 63, 'CMCA': 64, 'CMMM': 65, 'CCCM': 66,
                          'CMMA': 67, 'CCCA': 68, 'FMMM': 69, 'FDDD': 70, 'IMMM': 71, 'IBAM': 72, 'IBCA': 73,
                          'IMMA': 74, 'P4': 75, 'P41': 76, 'P42': 77, 'P43': 78, 'I4': 79, 'I41': 80, 'P-4': 81,
                          'I-4': 82, 'P4/M': 83, 'P42/M': 84, 'P4/N': 85, 'P42/N': 86, 'I4/M': 87, 'I41/A': 88,
                          'P422': 89, 'P4212': 90, 'P4122': 91, 'P41212': 92, 'P4222': 93, 'P42212': 94, 'P4322': 95,
                          'P43212': 96, 'I422': 97, 'I4122': 98, 'P4MM': 99, 'P4BM': 100, 'P42CM': 101, 'P42NM': 102,
                          'P4CC': 103, 'P4NC': 104, 'P42MC': 105, 'P42BC': 106, 'I4MM': 107, 'I4CM': 108, 'I41MD': 109,
                          'I41CD': 110, 'P-42M': 111, 'P-42C': 112, 'P-421M': 113, 'P-421C': 114, 'P-4M2': 115,
                          'P-4C2': 116, 'P-4B2': 117, 'P-4N2': 118, 'I-4M2': 119, 'I-4C2': 120, 'I-42M': 121,
                          'I-42D': 122, 'P4/MMM': 123, 'P4/MCC': 124, 'P4/NBM': 125, 'P4/NNC': 126, 'P4/MBM': 127,
                          'P4/MNC': 128, 'P4/NMM': 129, 'P4/NCC': 130, 'P42/MMC': 131, 'P42/MCM': 132, 'P42/NBC': 133,
                          'P42/NNM': 134, 'P42/MBC': 135, 'P42/MNM': 136, 'P42/NMC': 137, 'P42/NCM': 138, 'I4/MMM': 139,
                          'I4/MCM': 140, 'I41/AMD': 141, 'I41/ACD': 142, 'P3': 143, 'P31': 144, 'P32': 145, 'R3': 146,
                          'P-3': 147, 'R-3': 148, 'P312': 149, 'P321': 150, 'P3112': 151, 'P3121': 152, 'P3212': 153,
                          'P3221': 154, 'R32': 155, 'P3M1': 156, 'P31M': 157, 'P3C1': 158, 'P31C': 159, 'R3M': 160,
                          'R3C': 161, 'P-31M': 162, 'P-31C': 163, 'P-3M1': 164, 'P-3C1': 165, 'R-3M': 166, 'R-3C': 167,
                          'P6': 168, 'P61': 169, 'P65': 170, 'P62': 171, 'P64': 172, 'P63': 173, 'P-6': 174,
                          'P6/M': 175, 'P63/M': 176, 'P622': 177, 'P6122': 178, 'P6522': 179, 'P6222': 180,
                          'P6422': 181, 'P6322': 182, 'P6MM': 183, 'P6CC': 184, 'P63CM': 185, 'P63MC': 186,
                          'P-6M2': 187, 'P-6C2': 188, 'P-62M': 189, 'P-62C': 190, 'P6/MMM': 191, 'P6/MCC': 192,
                          'P63/MCM': 193, 'P63/MMC': 194, 'P23': 195, 'F23': 196, 'I23': 197, 'P213': 198, 'I213': 199,
                          'PM-3': 200, 'PN-3': 201, 'FM-3': 202, 'FD-3': 203, 'IM-3': 204, 'PA-3': 205, 'IA-3': 206,
                          'P432': 207, 'P4232': 208, 'F432': 209, 'F4132': 210, 'I432': 211, 'P4332': 212, 'P4132': 213,
                          'I4132': 214, 'P-43M': 215, 'F-43M': 216, 'I-43M': 217, 'P-43N': 218, 'F-43C': 219,
                          'I-43D': 220, 'PM-3M': 221, 'PN-3N': 222, 'PM-3N': 223, 'PN-3M': 224, 'FM-3M': 225,
                          'FM-3C': 226, 'FD-3M': 227, 'FD-3C': 228, 'IM-3M': 229, 'IA-3D': 230}
    Groupnum = Groupname_Groupnum[Groupname]
    return Groupnum


def Groupnum2CrystalSystem(Groupnum):
    CrystalSystem = ''
    if 0 < Groupnum <= 2:
        CrystalSystem = 'triclinic system'
    elif 3 <= Groupnum <= 15:
        CrystalSystem = 'monoclinic system'
    elif 16 <= Groupnum <= 74:
        CrystalSystem = 'orthorhombic system'
    elif 75 <= Groupnum <= 142:
        CrystalSystem = 'tetragonal system'
    elif 143 <= Groupnum <= 167:
        CrystalSystem = 'trigonal system'
    elif 168 <= Groupnum <= 194:
        CrystalSystem = 'hexagonal system'
    elif 195 <= Groupnum <= 230:
        CrystalSystem = 'cubic system'
    return CrystalSystem


def GetSymmetryEquivPosAsXyz(Groupname):
    def CalculateX(x, matrix):
        if x  == 'x':
            i = 0
        elif x == 'y':
            i = 4
        elif x == 'z':
            i = 8
        else:
            print('Input error')
            return

        if matrix[i] == 0:
            xpart = ''
        elif matrix[i] == 1.0:
            xpart = 'x'
        elif matrix[i] == -1.0:
            xpart = '-x'
        else:
            xpart = str(matrix[i]) + 'x'

        if matrix[i + 1] == 0:
            ypart = ''
        elif matrix[i + 1] == 1.0:
            ypart = 'y'
        elif matrix[i + 1] == -1.0:
            ypart = '-y'
        else:
            ypart = str(matrix[i + 1]) + 'y'

        if matrix[i + 2] == 0:
            zpart = ''
        elif matrix[i + 2] == 1.0:
            zpart = 'z'
        elif matrix[i + 2] == -1.0:
            zpart = '-z'
        else:
            zpart = str(matrix[i + 2]) + 'z'

        if matrix[i + 3] == 0:
            cpart = ''
        else:
            cpart = str(matrix[i + 3])

        SymmetryEquivPosAsXyz = [xpart, ypart, zpart, cpart]
        while '' in SymmetryEquivPosAsXyz:
            SymmetryEquivPosAsXyz.remove('')
        SymmetryEquivPosAsXyz = "+".join(SymmetryEquivPosAsXyz)
        return SymmetryEquivPosAsXyz


    wyckoffs, wyckoff_matrices = GetMatrix(Groupname)
    SymmetryEquivPosAsXyzs = []
    for matrix in wyckoff_matrices[wyckoffs[0]]:
        # print(matrix)
        X = CalculateX('x', matrix)
        Y = CalculateX('y', matrix)
        Z = CalculateX('z', matrix)
        SymmetryEquivPosAsXyz = X + ',' + Y + ',' + Z
        # print(SymmetryEquivPosAsXyz)
        SymmetryEquivPosAsXyzs.append(SymmetryEquivPosAsXyz)
    return SymmetryEquivPosAsXyzs


def CalculateVolume(abc):
    cosa2 = math.cos(math.radians(float(abc[3]))) * math.cos(math.radians(float(abc[3])))
    cosb2 = math.cos(math.radians(float(abc[4]))) * math.cos(math.radians(float(abc[4])))
    cosc2 = math.cos(math.radians(float(abc[5]))) * math.cos(math.radians(float(abc[5])))
    cosacosbcosc = math.cos(math.radians(float(abc[3]))) * math.cos(math.radians(float(abc[4]))) * math.cos(
        math.radians(float(abc[5])))
    v = round((float(abc[0]) * float(abc[1]) * float(abc[2]) * math.sqrt(
        1 - cosa2 - cosb2 - cosc2 + 2 * cosacosbcosc)) / 1000, 3)
    return v


def CalculateDistance2(coordinate1, coordinate2, retain = 6):
    x1 = coordinate1[0]
    y1 = coordinate1[1]
    z1 = coordinate1[2]

    x2 = coordinate2[0]
    y2 = coordinate2[1]
    z2 = coordinate2[2]

    distance2 = round((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2, retain)
    return distance2


def CalculateDistance(coordinate1, coordinate2, retain = 6):
    x1 = coordinate1[0]
    y1 = coordinate1[1]
    z1 = coordinate1[2]

    x2 = coordinate2[0]
    y2 = coordinate2[1]
    z2 = coordinate2[2]

    distance = round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2), retain)
    return distance


def GetMatrix(spaceGroupName, WycFilePath = '/'.join(sys.argv[0].replace('\\','/').split('/')[:-1]) +'/'+'wyckfull.dat'):
    # print(sys.argv[0])
    # SpaceGroupName = SpaceGroupName.upper()
    file = open(WycFilePath, 'r')
    wyckoffs = []
    wyckoff_matrices = {}
    while True:
        line = file.readline()
        if not line:
            break
        if line.strip() == spaceGroupName:
            file.readline()
            WyckoffNumber = int(file.readline())
            for i in range(WyckoffNumber):
                line = file.readline()
                wyckoff = line.split()[1]
                matrices = []
                multiplicity = int(line.split()[0])
                for i in range(multiplicity):
                    line = file.readline().split()
                    matrix = [float(i) for i in line]
                    # Matrix = line.split()
                    matrices.append(matrix)
                wyckoffs.append(wyckoff)
                wyckoff_matrices[wyckoff] = matrices
    file.close()
    return wyckoffs, wyckoff_matrices


def GetGeneralMatrix(spaceGroupName, WycFilePath = '/'.join(sys.argv[0].replace('\\','/').split('/')[:-1]) +'/'+'wyckfull.dat'):
    # print(sys.argv[0])
    # SpaceGroupName = SpaceGroupName.upper()
    file = open(WycFilePath, 'r')
    while True:
        line = file.readline()
        if not line:
            break
        if line.strip() == spaceGroupName:
            file.readline()
            file.readline()
            for i in range(1):
                line = file.readline()
                matrices = []
                multiplicity = int(line.split()[0])
                for i in range(multiplicity):
                    line = file.readline().split()
                    matrix = [float(i) for i in line]
                    # Matrix = line.split()
                    matrices.append(matrix)
    file.close()
    return matrices


def AddBoundaryAtom(coordinates):
    AfterCoordinate = []
    for coordinate in coordinates:
        AfterCoordinate.append(coordinate)
        if not 0 in coordinate:
            continue
        else:
            if coordinate[0] == 0 and coordinate[1] == 0 and coordinate[2] == 0:
                AfterCoordinate.append([0.0, 0.0, 1.0])
                AfterCoordinate.append([0.0, 1.0, 0.0])
                AfterCoordinate.append([0.0, 1.0, 1.0])
                AfterCoordinate.append([1.0, 0.0, 0.0])
                AfterCoordinate.append([1.0, 0.0, 1.0])
                AfterCoordinate.append([1.0, 1.0, 0.0])
                AfterCoordinate.append([1.0, 1.0, 1.0])
            elif coordinate[0] == 0 and coordinate[1] == 0:
                AfterCoordinate.append([0.0, 1.0, coordinate[2]])
                AfterCoordinate.append([1.0, 0.0, coordinate[2]])
                AfterCoordinate.append([1.0, 1.0, coordinate[2]])
            elif coordinate[0] == 0 and coordinate[2] == 0:
                AfterCoordinate.append([0.0, coordinate[1], 1.0])
                AfterCoordinate.append([1.0, coordinate[1], 0.0])
                AfterCoordinate.append([1.0, coordinate[1], 1.0])
            elif coordinate[1] == 0 and coordinate[2] == 0:
                AfterCoordinate.append([coordinate[1], 0.0, 1.0])
                AfterCoordinate.append([coordinate[1], 1.0, 0.0])
                AfterCoordinate.append([coordinate[1], 1.0, 1.0])
            elif coordinate[0] == 0:
                AfterCoordinate.append([1.0, coordinate[1], coordinate[2]])
            elif coordinate[1] == 0:
                AfterCoordinate.append([coordinate[0.0], 1.0, coordinate[2]])
            elif coordinate[2] == 0:
                AfterCoordinate.append([coordinate[0.0], coordinate[1], 1.0])
    return AfterCoordinate


def UniqueAtom2AllAtom(coordinates, spaceGroupName, retain = 6):
    allCoordinates = []
    matrices = GetGeneralMatrix(spaceGroupName)
    for coordinate in coordinates:
        allCoordinates = allCoordinates + SymmetricOperation(coordinate, matrices)
    return allCoordinates


def SymmetricOperation(coordinate, matrices, retain = 6):
    Coordinates = []
    for matrix in matrices:
        AfterCoordinate = Transformation(coordinate, matrix, retain)
        if not CoordinateInCoordinates(AfterCoordinate, Coordinates):
            Coordinates.append(AfterCoordinate)
    return Coordinates


def Transformation(BeforeCoordinate, matrix, retain = 6):
    def DealAfterCoordinate(Coordinate):

        x = Coordinate[0]
        y = Coordinate[1]
        z = Coordinate[2]
        while (x < 0):
            x = x + 1
        while (x >= 1):
            x = x - 1

        while (y < 0):
            y = y + 1
        while (y >= 1):
            y = y - 1

        while (z < 0):
            z = z + 1
        while (z >= 1):
            z = z - 1
        x = round(x, retain)
        y = round(y, retain)
        z = round(z, retain)
        return [x, y, z]

    x0 = BeforeCoordinate[0]
    y0 = BeforeCoordinate[1]
    z0 = BeforeCoordinate[2]
    x = x0 * matrix[0] + y0 * matrix[1] + z0 * matrix[2] + matrix[3]
    y = x0 * matrix[4] + y0 * matrix[5] + z0 * matrix[6] + matrix[7]
    z = x0 * matrix[8] + y0 * matrix[9] + z0 * matrix[10] + matrix[11]

    AfterCoordinate = [x, y, z]
    AfterCoordinate = DealAfterCoordinate(AfterCoordinate)
    # print(AfterCoordinate)
    # print()
    return AfterCoordinate


def Compare2Coordinate(coordinate1, coordinate2, retain = 4):
    accuracy = 2*10**(-retain)
    if  abs(coordinate1[0] - coordinate2[0]) <= accuracy and \
        abs(coordinate1[1] - coordinate2[1]) <= accuracy and \
        abs(coordinate1[2] - coordinate2[2]) <= accuracy:
        return True
    return False


def CoordinateInCoordinates(coordinate1, coordinates, retain = 4):
    for coordinate2 in coordinates:
        if Compare2Coordinate(coordinate1, coordinate2, retain):
            return True
    return False


def Compare2Coordinates(Coordinates1, Coordinates2, independent = False):
    for coordinate in Coordinates1:
        if not CoordinateInCoordinates(coordinate, Coordinates2):
            return False

    if not independent:
        for coordinate in Coordinates2:
            if not CoordinateInCoordinates(coordinate, Coordinates1):
                return False
    return True


if __name__ == '__main__':
    # coordinate = [0, 0, 0]
    # CellParameter = [10, 10, 10, 90, 90, 120]
    # print(Fractional2Cartesian(coordinate, CellParameter, retain=4))
    # GetSymmetryEquivPosAsXyz('IM-3M')
    pass
