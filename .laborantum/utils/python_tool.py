import json
import subprocess
import argparse
import os

def generateJson(answer):
  answerDataArray = []

  for index, answerData in enumerate(answer, start=1):
    answerDataArray.append({"testRes": answerData.__str__()})

  return json.dumps({"answer": answerDataArray})

def saveJson(answer, answerFolder):
  with open( answerFolder + '/.answer.json', 'w') as f:
    f.write(answer)

def answer(answer, answerFolder='.'):
  jsonFile = generateJson(answer)
  saveJson(jsonFile, answerFolder)

def runTask(taskPath):
  subprocess.run(["python",  taskPath])


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='Python tool', description='Tool for run and print code task.')
  parser.add_argument('taskPath')

  args = parser.parse_args()

  runTask(args.taskPath)