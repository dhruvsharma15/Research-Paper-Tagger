package com.sid.grobid.test;

import org.grobid.core.data.*;
import org.grobid.core.engines.config.GrobidAnalysisConfig;
import org.grobid.core.factory.*;
import org.grobid.core.main.GrobidHomeFinder;
import org.grobid.core.utilities.*;
import org.grobid.core.engines.Engine;

import java.io.*;
import java.util.Arrays;

public class test {

  public static void main(String[] args){

      final File fpath = new File("D:\\Study Material\\M.S\\FALL 18\\Data Analytics\\Project\\files");
      
      String pGrobidHome = "C:\\grobid-0.5.1\\grobid-0.5.1\\grobid-home";

      // The GrobidHomeFinder can be instantiate without parameters to verify the grobid home in the standard
      // location (classpath, ../grobid-home, ../../grobid-home)

      // If the location is customised:
      GrobidHomeFinder grobidHomeFinder = new GrobidHomeFinder(Arrays.asList(pGrobidHome));

      //The GrobidProperties needs to be instantiate using the correct grobidHomeFinder or it will use the default
      //locations
      GrobidProperties.getInstance(grobidHomeFinder);

      System.out.println(">>>>>>>> GROBID_HOME="+GrobidProperties.get_GROBID_HOME_PATH());

      Engine engine = GrobidFactory.getInstance().createEngine();

      // Biblio object for the result
      BiblioItem resHeader = new BiblioItem();
      
      for (final File fileEntry : fpath.listFiles())
      {
      
      //fpath = fpath+i+".pdf";

      test pdg = new test();

      try {
          String tei = engine.fullTextToTEI(fileEntry, GrobidAnalysisConfig.builder().build());

          pdg.write(tei,fileEntry.getName());

          System.out.println("done");

      }
      catch (Exception e) {
          // If an exception is generated, print a stack trace
    	  System.out.println("error");
          e.printStackTrace();
      }
      }
  }

  public void write(String tei, String fname) throws IOException {
	  //ofpath = ofpath+i+".xml";
	  
	  String[] file_name = fname.split("\\.");
	  
	  System.out.println(fname);
	  
	  System.out.println(file_name[0]);
	  
	  String ofpath = new String("D:\\Study Material\\M.S\\FALL 18\\Data Analytics\\Project\\XML_Files\\");
	  
	  ofpath = ofpath+file_name[0]+".xml";
	  
	  File fout = new File(ofpath);
      FileOutputStream fos = new FileOutputStream(fout);

      BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

      System.out.println("writing "+fname);

      bw.write(tei);

      System.out.println("done writing "+fname);
      
      
      BufferedWriter bw2 = null;
      bw2 = new BufferedWriter(new FileWriter("D:\\Study Material\\M.S\\FALL 18\\Data Analytics\\Project\\indexes.txt", true));
 	  bw2.write(file_name[0]);
 	  bw2.newLine();
 	  bw2.flush();

      bw.close();
      bw2.close();

  }


}
