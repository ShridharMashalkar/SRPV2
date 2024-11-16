import subprocess

def main():
    try:
        subprocess.run(["python", "DC_ColoredMNISTTest.py"])
        subprocess.run(["python", "DC_ColoredMNISTTrain.py"])
        subprocess.run(["python", "DC_NormalMNISTTest.py"])
        subprocess.run(["python", "DC_NormalMNISTTrain.py"])
    except Exception as e: 
        print(e)
        pass

if __name__ == '__main__':
    main()


