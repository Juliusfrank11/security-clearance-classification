
import os
import re
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

def get_FoF_and_formal_findings_text(path):
    lines = [l.strip() for l in extract_text(path).split("\n") if l.strip()]
    finding_of_fact = ""
    formal_findings = ""
    scanning_FoF = False
    scanning_formal_findings = False
    for line in lines:
        if line not in "123456798": #skip page numbers
          alphabetic_only_line = re.sub(r"[^A-Za-z\s]+", "", line).strip()
          if alphabetic_only_line == "Findings of Fact":
              scanning_FoF = True
              continue
          if alphabetic_only_line == "Analysis" or alphabetic_only_line == "Policies":
              scanning_FoF = False
          if alphabetic_only_line == "Formal Findings":
              scanning_formal_findings = True
              continue
          if scanning_FoF:
              finding_of_fact += line
              finding_of_fact += "\n"
          if scanning_formal_findings:
              formal_findings += line
              formal_findings += "\n"
    return finding_of_fact, formal_findings


dirs_to_make = (
    r"data/finding_of_fact/",
    r"data/formal_findings/"
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
                    rf"data\finding_of_fact\{hearing_pdf_file_name.replace('.pdf', '.txt')}"
                ),
                os.path.exists(
                    rf"data\formal_findings\{hearing_pdf_file_name.replace('.pdf', '.txt')}"
                ),
            ]
        ):
            finding_of_fact, formal_findings = get_FoF_and_formal_findings_text(path)
            # Cases ending in approval always have the string "is granted" in their formal_findingss
            # As part of "access to classified infomation *is granted*" or similar
            with open(
                rf"data\finding_of_fact\{hearing_pdf_file_name.replace('.pdf', '.txt')}",
                "w",
                errors="ignore",
                encoding="utf-8",
            ) as f:
                f.write(finding_of_fact)
            with open(
                rf"data\formal_findings\{hearing_pdf_file_name.replace('.pdf', '.txt')}",
                "w",
                errors="ignore",
                encoding="utf-8",
            ) as f:
                f.write(formal_findings)
    except PDFSyntaxError:
        pass


def parallel_make_txt_data_files_for_hearings():
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(make_txt_data_file_for_hearing, os.listdir(r"data/hearings"))

def process_formal_findings(formal_finding_path):
  def get_guideline_letter(line: str) -> str:
    return re.findall(r"Guideline\W+[A-M]",line,flags=re.IGNORECASE)[0].split()[-1].upper()
  def get_allogation_result(line: str) -> str:
    if "for" in line.lower().split():
      return True #"Accepted"
    elif "against" in line.lower().split():
      return False#"Denied"
    else:
      return np.nan#"Withdrawn" or not listed

    
  with open(formal_finding_path,"r",encoding="utf-8") as f:
    lines = [l.strip().lower() for l in f.readlines()]
  
  assert lines, f"no text in {formal_finding_path}"

  d = {}
  i_conclusions_already_assigned = []
  try:
    conclusion_index = lines.index("conclusion")
  except ValueError:
    conclusion_index = lines.index("conclusions")
  for n, k in enumerate(lines[:conclusion_index]):
    if "paragraph" in k:
      for m, v in enumerate(lines[n:conclusion_index]):
        if ("applicant" in v) and not (m in i_conclusions_already_assigned):
          if "guideline" in k.lower():
            try:
              d[get_guideline_letter(k)] = get_allogation_result(v)
            except IndexError:
              d[k] = get_allogation_result(v)
          else:
            pass # handle subparagraphs later
          i_conclusions_already_assigned.append(m)
          break
  return d



if __name__ == "__main__":
  parallel_make_txt_data_files_for_hearings()


  all_formal_finding_results = {}

  for file in  os.listdir(r"data\formal_findings"):
    try:
      with open(fr"data\finding_of_fact\{file}","r",encoding="utf-8") as f:
        text = f.read()
      all_formal_finding_results[text] = process_formal_findings(fr"data\formal_findings\{file}")
    except:
      print(file)
  
  results = pd.DataFrame(all_formal_finding_results).T.drop(['subparagraph 3.a: paragraph 4, guideline',
       'paragraph  1, guideline  1:', 'paragraph  1, guideline:'],axis=1)
  
  for col in results.columns:
    results[col].rename("label").dropna().to_csv(f"data/formal_finding_results_guideline_{col}.csv",index_label="text",columns=["label"])