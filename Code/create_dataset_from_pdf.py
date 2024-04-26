from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
import os
from multiprocessing import Pool
import re
from concurrent.futures import ThreadPoolExecutor
from random import randint


def clean_text_in_parathesis(text):
    return re.sub(r"\((.*?)\)", "", text)


def get_FoF_and_conclusion_text(path):
    lines = [l.strip() for l in extract_text(path).split("\n") if l.strip()]
    finding_of_fact = ""
    conclusion = ""
    scanning_FoF = False
    scanning_conclusion = False
    for line in lines:
        alphabetic_only_line = re.sub(r"[^A-Za-z\s]+", "", line).strip()
        if alphabetic_only_line == "Findings of Fact":
            scanning_FoF = True
            continue
        if alphabetic_only_line == "Analysis" or alphabetic_only_line == "Policies":
            scanning_FoF = False
        if alphabetic_only_line == "Conclusion":
            scanning_conclusion = True
            continue
        if scanning_FoF:
            finding_of_fact += line
            finding_of_fact += " "
        if scanning_conclusion:
            conclusion += line
            conclusion += " "
    finding_of_fact = clean_text_in_parathesis(finding_of_fact)
    return finding_of_fact, conclusion


dirs_to_make = (
    r"data/finding_of_fact/train/approve",
    r"data/finding_of_fact/eval/approve",
    r"data/finding_of_fact/train/deny",
    r"data/finding_of_fact/eval/deny",
    r"data/conclusion/approve",
    r"data/conclusion/deny",
)
for d in dirs_to_make:
    if not os.path.exists(d):
        os.makedirs(d)


def make_txt_data_file_for_hearing(hearing_pdf_file_name, test_split=0.2):
    path = rf"data\hearings\{hearing_pdf_file_name}"
    try:
        print(path)
        if not any(
            [
                os.path.exists(
                    rf"data\finding_of_fact\eval\approve\{hearing_pdf_file_name.replace('.pdf', '.txt')}"
                ),
                os.path.exists(
                    rf"data\finding_of_fact\eval\deny\{hearing_pdf_file_name.replace('.pdf', '.txt')}"
                ),
                os.path.exists(
                    rf"data\finding_of_fact\train\approve\{hearing_pdf_file_name.replace('.pdf', '.txt')}"
                ),
                os.path.exists(
                    rf"data\finding_of_fact\train\deny\{hearing_pdf_file_name.replace('.pdf', '.txt')}"
                ),
            ]
        ):
            finding_of_fact, conclusion = get_FoF_and_conclusion_text(path)
            # Cases ending in approval always have the string "is granted" in their conclusions
            # As part of "access to classified infomation *is granted*" or similar
            classification = (
                "approve"
                if "isgranted" in "".join(conclusion.split()).lower()
                else "deny"
            )
            if randint(1, 100) > round(100 * (1 - test_split)):
                split = "eval"
            else:
                split = "train"
            with open(
                rf"data\finding_of_fact\{split}\{classification}\{hearing_pdf_file_name.replace('.pdf', '.txt')}",
                "w",
                errors="ignore",
                encoding="utf-8",
            ) as f:
                f.write(finding_of_fact)
            with open(
                rf"data\conclusion\{classification}\{hearing_pdf_file_name.replace('.pdf', '.txt')}",
                "w",
                errors="ignore",
                encoding="utf-8",
            ) as f:
                f.write(conclusion)
    except PDFSyntaxError:
        pass


def parallel_make_txt_data_files_for_hearings():
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(make_txt_data_file_for_hearing, os.listdir(r"data/hearings"))


parallel_make_txt_data_files_for_hearings()
