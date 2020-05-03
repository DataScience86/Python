from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
 
import glob
import textract
 
from gensim import utils
import pandas as pd
from simserver import SessionServer
import logging
import re
import string
import shutil
 
from Tkinter import *
import tkFileDialog
import Tkinter
 
#import PIL
 
#from PIL import ImageTk, Image
 

class Simple_resume_similarity_app_tk(Tkinter.Tk):
 
    def __init__(self):
        Tkinter.Tk.__init__(self)
        self.initialize()
 
    def initialize(self):
 
        button = Tkinter.Button(self,text=u"Click Me!", command = self.resume_scoring)
        button.grid(row = 1, column = 1)
 
 
        self.label1 = Tkinter.Label(self, text = "Click Button To Generate Similarity Score")
        self.label1.grid(row =  2, column = 1)
 
        #self.img = Image.open('C:\\temp\\Resume_Similarity\\Resume_GUI\\wellsfargologo2.gif')
        #self.img_path = r"C:/temp/Resume_Similarity/Resume_GUI/wellsfargologo2.gif"
        #self.im = Image.open(self.img_path)
        #self.ph = PIL.ImageTk.PhotoImage(self.im)
 
        #self.label1 = Label(self, image=self.ph)
        #self.label1.image = self.ph
        #self.label1.pack(side = "left")
 
        #logo = PhotoImage("C:/temp/Resume_Similarity/Resume_match_score/logo.jpg")
        #label.config(image = logo)
 
    def resume_scoring(self):
        """"
            Cleanes the data and runs the resume matching code. User is
            requested to pass the job description name, session_name and
            final output file name. Final output is an excel file.
 
            @param: job_description - string
            @param: session_name - string
            @param: output_filename - string
 
            Once you run this code it will prompt you to select the path of the
            directory           
        """
 
 
        self.job_description = self.select_job_description()
        if len(self.job_description) > 0:
 
            #self.job_description_path = os.path.join( self.job_description_path + "/" + job_description)
 
            self.raw_resumes_path =self.select_resume_path()
            if len(self.raw_resumes_path) > 0:               
                self.save_text_files_path = self.select_rawtext_path()
 
                self.raw_resumes_to_text()
                self.jd_to_text()
 
                self.file_list_text = glob.glob(self.save_text_files_path + "/*.*")
                print self.file_list_text
 
                self.resume_id = []
                for i in range(0, len(self.file_list_text)):
                    self.resume_id.append([int(s) for s in self.file_list_text[i].split() if s.isdigit()])
 
                self.documents = []
                for filename in self.file_list_text:
                    with open(filename, 'r') as f:
                        #d = f.read()
                        #print d
                        self.documents.append(f.read())
 
                self.corpus = [{'id': 'doc_%s' % num, 'tokens': utils.simple_preprocess(text)}
                  for num, text in enumerate(self.documents)]
 
                self.count = 0
                while self.count < len(self.resume_id):   
                    for item in self.corpus:
                        if self.resume_id[self.count] == []:
                            item['id'] = 'doc_jd'
                        else:
                            item['id'] = str(self.resume_id[self.count])
                        self.count =  self.count + 1
 
                self.regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html
                self.tokenized_corpus_no_punctuation = []
 
                for review in self.corpus:       
                    self.new_corpus = []
                    for token in review:
                        self.new_token = self.regex.sub(u'', token)
                        if not self.new_token == u'':
                            self.new_corpus.append(self.new_token)       
                    self.tokenized_corpus_no_punctuation.append(self.new_corpus)
 
                self.dir_name = self.setting_up_server_session_dir()       
                self.server = SessionServer(self.dir_name)
 
                logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
                self.server.train(self.corpus, method='lsi')
                self.server.index(self.corpus)
                self.lst = self.server.find_similar('doc_jd')
                self.series = pd.DataFrame(self.lst)
                self.series.columns = ['Resume_ID', 'Score', 'none']
                self.series.index.names = ['Rank']
 
                self.series = self.series.drop(self.series.columns[2], axis = 1)       
                self.final_excel_path()
 
 
    def setting_up_server_session_dir(self):
        self.dir = 'C:/temp/resume_server_script_server_logs'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        else:
            shutil.rmtree(self.dir)           #removes all the subdirectories!
            os.makedirs(self.dir)
        return self.dir
 
 
    def convert(self,fname, pages=None):
        if not pages:
            pagenums = set()
        else:
            pagenums = set(pages)         
        output = StringIO()       
        manager = PDFResourceManager()       
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)       
        infile = file(fname, 'rb')       
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        text = output.getvalue()
        output.close
        return text
 
    def select_job_description(self):
        root = Tkinter.Tk()
        root.withdraw() #use to hide tkinter window
 
        currdir = os.getcwd()
        self.tempdir = tkFileDialog.askopenfilename(parent=root,
                                            initialdir=currdir,
                                            title="Select Job Description file")
        if len(self.tempdir) > 0:
            return self.tempdir
 
    def select_resume_path(self):
        root = Tkinter.Tk()
        root.withdraw() #use to hide tkinter window
 
        currdir = os.getcwd()
        tempdir = tkFileDialog.askdirectory(parent=root,
                                            initialdir=currdir,
                                            title="Select Resume Description Path")
        if len(tempdir) > 0:
            return tempdir
 
    def select_rawtext_path(self):
        root = Tkinter.Tk()
        root.withdraw() #use to hide tkinter window
 
        currdir = os.getcwd()
        tempdir = tkFileDialog.askdirectory(parent=root,
                                            initialdir=currdir,
                                            title="Select Path Where You Want To Save Text Files.")
        if len(tempdir) > 0:
            return tempdir
 
    def final_excel_path(self):
        root = Tkinter.Tk()
        root.withdraw() #use to hide tkinter window
 
        currdir = os.getcwd()
        savefile  = tkFileDialog.asksaveasfilename(filetypes=(("Excel files", "*.xlsx"),
                                                         ("All files", "*.*") ),
                                            parent=root,
                                            initialdir=currdir,
                                            title="Final Excel Output Path")
        if len(savefile) > 0:
            self.series.to_excel(savefile + ".xlsx", index=True, sheet_name="Results")        
 
 
 
    def raw_resumes_to_text(self):
        ## Reading the files path
        file_list_raw = glob.glob(self.raw_resumes_path + "/*.*")
        for fp in file_list_raw:
        # print fp
            ext = os.path.splitext(fp)[-1].lower()
            base = os.path.basename(fp)
            file_name = os.path.splitext(base)[0]   
 
            #print ext
            if ext == ".docx":       
                text = textract.process(fp)
                complte_name = os.path.join(self.save_text_files_path + "/" + file_name + ".txt")        
                with open(complte_name, 'w') as f:
                    f.write(text)
 
            elif ext == ".pdf":       
                text = self.convert(fp)
                complte_name = os.path.join(self.save_text_files_path + "/" + file_name + ".txt")        
                with open(complte_name, 'w') as f:
                    f.write(text)
            elif ext == ".txt":
                shutil.copy(os.path.join(self.raw_resumes_path + str("/") + file_name + ".txt"), os.path.join(self.save_text_files_path + str("/") + file_name + ".txt"))
            else:       
                print "Unable to recognise this format."
 
    def jd_to_text(self):
 
        ext = os.path.splitext(self.job_description)[-1].lower()
        file_name_with_ext = os.path.basename(self.job_description)
        file_name = os.path.splitext(file_name_with_ext)[0].lower()
 
        if ext == ".docx":
            text = textract.process(self.job_description)
            complte_name = os.path.join(self.save_text_files_path + "/" + file_name + ".txt")        
            with open(complte_name, 'w') as f:
                f.write(text)
 
        elif ext == ".pdf":       
            text = convert(self.job_description)
            complte_name = os.path.join(self.save_text_files_path + "/" + file_name + ".txt")        
            with open(complte_name, 'w') as f:
                f.write(text)
 
        elif ext == ".txt":
            shutil.copy(self.job_description, os.path.join(self.save_text_files_path + str("/") + file_name + ".txt"))
 
        else:
            print "This file format is not supported for now."
 
 
 
if __name__ == "__main__":
    app = Simple_resume_similarity_app_tk()
    app.title('Similarity Score Generator')
    app.mainloop()
