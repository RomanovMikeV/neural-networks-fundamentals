import argparse
import math
import numbers
import numpy
import os
import Levenshtein
# import torch

from json_tricks import load, dump
from copy import deepcopy
from pathlib import Path
from nbconvert.exporters import PythonExporter


def check_condition(condition, info, successes, failures, test_weight):
    if condition:
        successes.append((info, 'PASSED', test_weight))
    else:
        failures.append((info, 'FAILED', test_weight))


def get(collection, key):
    res = None
    message = None
    try:
        res = collection[key]
    except:
        res = None

    return res


def run_checks(
        received,
        expected,
        test_info='',
        eps=1.0e-4):

    report = []

    if expected is None:
        if received is None:
            explanation = "OK"
            return  [{'name': test_info, 'ok': True, 'status': explanation}]
        else:
            explanation = f"Expecting None, got {str(received)}"
            return [{'name': test_info, 'ok': False, 'status': explanation}]


    if isinstance(expected, dict):
        report = []

        for key in expected:
            new_expected = get(expected, key)
            new_received = get(received, key)

            test_info_new = test_info + ' -> ' + str(key)

            report.extend(run_checks(new_received, new_expected, test_info_new))
        return report

    if isinstance(expected, str):
        if not isinstance(received, str):
            explanation = "FAILED: String is expected"
            return [{'name': test_info, 'ok': False, 'status': explanation}]

        misfit = Levenshtein.distance(expected, received)
        if misfit != 0:
            explanation = "FAILED: Edit distance between strings is " + str(misfit)
            return [{'name': test_info, 'ok': False, 'status': explanation}]
        else:
            explanation = 'OK'
            return [{'name': test_info, 'ok': True, 'status': explanation}]


    if isinstance(expected, (list, tuple)):
        report = []

        for index in range(len(expected)):
            new_expected = get(expected, index)
            new_received = get(received, index)

            test_info_new = test_info + ' -> ' + str(index)
            one_check = run_checks(new_received, new_expected, test_info_new)
            report.extend(one_check)
        return report


    if isinstance(expected, numbers.Number):
        try:
            misfit = abs(expected - received)
            check = (misfit < eps)
            explanation = 'OK'
            if not check:
                explanation = '''
                    FAILED: Incorrect answer. Misfit is {},'
                    that is larger than eps ({})'''.format(misfit, eps)
            return [{'name': test_info, 'ok': check, 'status': explanation}]

        except:
            explanation = 'FAILED: The answer is not found or is of incorrect Type'
            return [{'name': test_info, 'ok': False, 'status': explanation}]


    if isinstance(expected, numpy.ndarray):

        # Checking if the answer exists
        if received is None:
            explanation = 'FAILED. Item is not found'
            return [{'name': test_info, 'ok': False, 'status': explanation}]

        if not isinstance(received, numpy.ndarray):
            explanation = f'FAILED. Expected numpy array, received {type(received)} instead'
            return [{'name': test_info, 'ok': False, 'status': explanation}]

        # Checking if Numpy array is empty
        if expected.size == 0:
            condition = (received.size == 0)
            if not condition:
                explanation = (
                    'FAILED: expected an empty tensor, got non-empty one:\n' +
                    'Expected size: {}\n'.format(str(expected.size)) +
                    'received size: {}\n'.format(str(received.size))
                )
                return [{'name': test_info, 'ok': False, 'status': explanation}]

        # Checking if Numpy arrays have the same dtype
        if received.dtype != expected.dtype:
            explanation = (
                'FAILED: data types do not match:\n' +
                'Expected type: {}'.format(expected.dtype) + '\n' +
                'received type: {}'.format(received.dtype)
            )
            return [{'name': test_info, 'ok': False, 'status': explanation}]

        # Checking if the arrays have the same number of dimensions
        if len(expected.shape) != len(received.shape):
            explanation = (
                'FAILED: tensor dimensionalities do not match:\n' +
                'Expected shape: {}\n'.format(expected.shape) +
                'received shape: {}'.format(received.shape)
            )
            return [{'name': test_info, 'ok': False, 'status': explanation}]

        # Checking if the arrays have the same shape
        if expected.shape != received.shape:
            explanation = (
                'FAILED: tensor shapes do not match:\n' +
                'Expected shape: {}\n'.format(expected.shape) +
                'received shape: {}'.format(received.shape)
            )
            return [{'name': test_info, 'ok': False, 'status': explanation}]

        # Checking the arrays that have the exact types
        if received.dtype in [numpy.bool_, numpy.byte, numpy.ubyte]:
            if not (received == expected).all():
                explanation = (
                    'FAILED: tensor shapes do not match:\n' +
                    'Expected shape: {}\n'.format(expected.shape) +
                    'received shape: {}'.format(received.shape)
                )
                return [{'name': test_info, 'ok': False, 'status': explanation}]

        # Checking inexact typed arrays
        else:
            mismatch = numpy.abs((received - expected))
            avg_mismatch = mismatch.mean()
            if not avg_mismatch < eps:
                explanation = (
                    'FAILED: answers do not match:\n' +
                    'Maximal average mismatch: {}\n'.format(str(eps)) +
                    'received average mismatch: {}'.format(str(avg_mismatch))
                )
                return [{'name': test_info, 'ok': False, 'status': explanation}]

        return [{'name': test_info, 'ok': True, 'status': 'OK'}]

    # if isinstance(expected, torch.Tensor):
    #     return run_checks(
    #         received.detach().numpy(),
    #         expected.detach().numpy(),
    #         eps=eps)


def replace_last_entrance(path, to_replace, replacement):
    src_parts = list(path.parts[::-1])
    replace_index = src_parts.index(to_replace)
    if replace_index > 0:
        src_parts[replace_index] = replacement
    src_parts = src_parts[::-1]
    return Path("/".join(src_parts))


def get_teacher_path(task_path, fname='.solution.json'):
    teacher_path = Path(task_path) / fname
    # teacher_path = replace_last_entrance(
        # task_path, 'texts', 'solutions') / fname
    return teacher_path


def get_student_path(task_path, fname='.answer.json'):
    student_path = Path(task_path) / fname
    # student_path = replace_last_entrance(
        # task_path, 'texts', 'answers') / fname
    return student_path


def get_report_path(task_path, fname='.report.json'):
    report_path = Path(task_path) / fname
    # report_path = replace_last_entrance(
        # task_path, 'texts', 'reports') / fname
    return report_path


def check_json(task_path):
    teacher_path = get_teacher_path(task_path)
    student_path = get_student_path(task_path)
    report_path = get_report_path(task_path)

    student_state = load(str(student_path))
    teacher_state = load(str(teacher_path))

    report = run_checks(student_state, teacher_state, test_info='root')
    report = {'test_cases': report}

    report_path.parent.mkdir(parents=True, exist_ok=True)

    dump(report, str(report_path))


def write_result(result, task_path):
    student_path = get_student_path(task_path)
    student_path.split('/')
    dump(result, student_path)


def ipynb_to_py(source, target):
    py_exporter = PythonExporter()
    source, meta = py_exporter.from_filename(source)

    Path(target).unlink(missing_ok=True)

    with open(target, 'wb') as fout:
        fout.write(source.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

    parser.add_argument('-t', '--task',
                        help="Task path",
                        type=Path)

    args = parser.parse_args()

    check_json(args.task)
