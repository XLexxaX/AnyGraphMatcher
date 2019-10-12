package de.uni_mannheim.informatik.dws.ontmatching.demomatcher;

import de.uni_mannheim.informatik.dws.ontmatching.demomatcher.SimpleStringMatcher.Blocker.Tuple;
import de.uni_mannheim.informatik.dws.ontmatching.matchingbase.OaeiOptions;
import de.uni_mannheim.informatik.dws.ontmatching.matchingjena.MatcherYAAAJena;
import de.uni_mannheim.informatik.dws.ontmatching.yetanotheralignmentapi.Alignment;
import de.uni_mannheim.informatik.dws.ontmatching.yetanotheralignmentapi.Correspondence;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.ProcessBuilder.Redirect;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import org.apache.jena.ontology.OntModel;
import org.apache.jena.ontology.OntResource;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.rdf.model.StmtIterator;
import org.apache.jena.riot.RDFDataMgr;
import org.apache.jena.riot.RDFFormat;
import org.apache.jena.util.FileManager;
import org.apache.jena.util.iterator.ExtendedIterator;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.complexPhrase.ComplexPhraseQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.Version;

/**
 * A simple string matcher using String equivalence as matching criterion.
 */
public class SimpleStringMatcher extends MatcherYAAAJena {

	@SuppressWarnings("unused")
	private final String NEWLINE = System.getProperty("line.separator");
	private final String BASEDIR = System.getProperty("user.home");
	private final String DSEP = File.separator;
	@SuppressWarnings("unused")
	private final String CWD = System.getProperty("user.dir");
	private final String AGMDIR = "oaei-resources" + DSEP + "AnyGraphMatcher";// "D:"+DSEP+"Development"+DSEP+"Code"+DSEP+"AnyGraphMatcher";
	private String last_msg = "";
	
	public boolean findFile(String name, String filepath, boolean withPostfix) {
		File file = new File(filepath);
		File[] list = file.listFiles();
		if (list != null)
			for (File fil : list) {
				String fname = fil.getName();
				if (!withPostfix && fname.contains("."))
					fname = String.join("", Arrays.copyOfRange(fname.split("\\."), 0, fname.split("\\.").length - 1));
				if (name.equalsIgnoreCase(fname)) {
					return true;
				}
			}
		return false;
	}

	private void logProgress(String msg) {
		if (!msg.equals(last_msg)) {
			last_msg = msg;
			System.err.print(msg);
		}
	}
	
	@Override
	public Alignment match(OntModel source, OntModel target, Alignment inputAlignment, Properties p) throws Exception {
		//if (!source.getGraph().toString().contains("darkscape") || !target.getGraph().toString().contains("oldschoolrunescape")) {
		//if (!source.getGraph().toString().contains("memoryalpha") || !target.getGraph().toString().contains("stexpanded")) {
		if (!source.getGraph().toString().toLowerCase().contains("marvelcinematic")) {
			System.out.println("Skipping ...");
			Alignment alignment = new Alignment();
			return alignment;
		}
		/*if 
		 * (!source.getGraph().toString().contains("m arvelcinematicuniverse") || !target.getGraph().toString().contains("marvel")) {
			System.out.println("Skipping ...");
			Alignment alignment = new Alignment();
			return alignment;
		}*/
		
		HashMap<String, String> cased_values = new HashMap<String, String>(); 
		this.logProgress("Starting matching\n");
		Alignment alignment = new Alignment();

		ProcessBuilder pb = new ProcessBuilder();
		Map<String, String> envs = pb.environment();
		String PYTHONDIR = "";
		String pythonname = "python";
		this.logProgress("You are unning on "+System.getProperty("os.name")+"\n");
		if (System.getProperty("os.name").toLowerCase().contains("linux")) {

			if (System.getenv("PATH") != null) {
				for (String pathVar2 : System.getenv("PATH").split(";")) {
					for (String pathVar : pathVar2.split(":")) {
						if (this.findFile("python3", pathVar, false)) {
							PYTHONDIR = pathVar;
							pythonname = "python3";
						} else if (this.findFile("python", pathVar, false)) {
							PYTHONDIR = pathVar;
						}
					}
				}

			}
		} else if (System.getProperty("os.name").toLowerCase().contains("windows")) {

			if (System.getenv("Path") != null) {
				for (String pathVar : System.getenv("Path").split(";")) {
					if (this.findFile("python3.exe", pathVar, true)) {
						PYTHONDIR = pathVar;
						pythonname = "python3";
					} else if (this.findFile("python.exe", pathVar, true)) {
						PYTHONDIR = pathVar;
					}
				}
			}
		} else {
			throw new Exception("Program can only run on Linux or Windows based systems!");
		}
		if (PYTHONDIR.equals("")) {
			throw new Exception("Python must be registered in the Path/PATH environment-variable");
		} else {
			this.logProgress("Python found in " + PYTHONDIR + "\n");
		}

		Blocker blocker = new Blocker();
		this.logProgress("-Processing source dataset-\n");
		triplize(source, "source", true, blocker, cased_values);
		this.logProgress("-Processing target dataset-\n");
		triplize(target, "target", false, blocker, cased_values);
		blocker = null;
		source = null;
		target = null;
		inputAlignment = null;
		p = null;
		System.gc();

		
		List<String> call_matcher_command = new ArrayList<>();
		call_matcher_command.add(PYTHONDIR + DSEP + pythonname);
		call_matcher_command.add("agm_cli_lite.py");
		call_matcher_command.add("-s");
		call_matcher_command.add("" + BASEDIR + DSEP + "oaei_track_cache" + DSEP + "tmpdata" + DSEP
				+ "graph_triples_source.nt" + "");
		call_matcher_command.add("-t");
		call_matcher_command.add("" + BASEDIR + DSEP + "oaei_track_cache" + DSEP + "tmpdata" + DSEP
				+ "graph_triples_target.nt" + "");
		call_matcher_command.add("-p");
		call_matcher_command.add(
				"" + BASEDIR + DSEP + "oaei_track_cache" + DSEP + "tmpdata" + DSEP + "possible_matches.csv" + "");
		call_matcher_command.add("-g");
		call_matcher_command.add("" + BASEDIR + DSEP + "oaei_track_cache" + DSEP + "tmpdata" + DSEP
				+ "artificial_gold_standard.csv" + "");

		
		pb.directory(new File(AGMDIR + DSEP + "cle"));

		pb.command(call_matcher_command);

		pb.redirectInput(Redirect.INHERIT); // no need because the process gets no further input than the process
											// parameters
		pb.redirectOutput(Redirect.INHERIT); // no need because we want to collect it
		pb.redirectError(Redirect.INHERIT); // redirect err pipe because of all logging etc

		// System.err.println("Start external matcher with command: " + String.join(" ",
		// activate_env_command));
		System.err.println("Start external matcher with command: " + String.join(" ", call_matcher_command));
		Process process = pb.start();

		int errCode = process.waitFor(); // wait for the matcher to finish
		if (errCode != 0) {
			System.err.println("Error code of external matcher is not equal to 0.");
		}

		StringBuilder sb = new StringBuilder();
		try (BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
			String line = null;
			while ((line = br.readLine()) != null) {
				System.err.print(line);
			}
		} catch(Exception e) {
			this.logProgress("Matching-process has thrown an exception: " + e.getMessage());
			e.printStackTrace();
		}

		Path path = Paths.get(AGMDIR + DSEP + "result_data" + DSEP + "cli_result" + DSEP + "married_matchings.csv");
		byte[] bytes = Files.readAllBytes(path);
		List<String> married_matchings_per_line = Files.readAllLines(path, StandardCharsets.UTF_8);
		for (String x : married_matchings_per_line) {
			String[] uris = x.split("\t");
			if (uris[0].length() > 0)
				alignment.add(uris[1], uris[2]);
		}
		
		return alignment;
	}

	@SuppressWarnings("unused")
	private void matchResources(ExtendedIterator<? extends OntResource> sourceResources,
			ExtendedIterator<? extends OntResource> targetResources, Alignment alignment) {
		HashMap<String, String> text2URI = new HashMap<>();
		while (sourceResources.hasNext()) {
			OntResource source = sourceResources.next();
			text2URI.put(getStringRepresentation(source), source.getURI());
		}
		while (targetResources.hasNext()) {
			OntResource target = targetResources.next();
			String sourceURI = text2URI.get(getStringRepresentation(target));
			if (sourceURI != null) {
				alignment.add(sourceURI, target.getURI());
			}
		}
	}

	private String getStringRepresentation(OntResource resource) {
		String arbitraryLabel = resource.getLabel(null);
		if (arbitraryLabel != null)
			return arbitraryLabel;
		return resource.getLocalName();
	}

	private void triplize(OntModel model, String datasetname, boolean indexed, Blocker blocker, HashMap<String, String> cased_values) {
		
		String folder = BASEDIR + DSEP + "oaei_track_cache" + DSEP + "tmpdata" + DSEP + "";
		File f = new File(folder + "graph_triples_" + datasetname + ".nt");
		File f2 = new File(folder + "possible_matches.csv");
		File f3 = new File(folder + "artificial_gold_standard.csv");
		BufferedWriter blockedwriter = null;
		BufferedWriter gswriter = null;
		
		File directory = new File(folder);
	    if (! directory.exists()){
	        directory.mkdirs();
	    }
		
		if (f.exists())
			f.delete();
		if (f2.exists())
			f2.delete();
		if (f3.exists())
			f3.delete();

		try {
			this.logProgress("\rTransforming data format: 0%\r");
			RDFDataMgr.write(new FileOutputStream(f), model, RDFFormat.NTRIPLES_UTF8);
			this.logProgress("Transforming data format: 100%\n");
		} catch (FileNotFoundException e) {
			this.logProgress("Data transformation has thrown an exception: " + e.getMessage());
			e.printStackTrace();
		}


		try {
			f2.createNewFile();
			f3.createNewFile();
			blockedwriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f2, true), StandardCharsets.UTF_8));
			gswriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f3, true), StandardCharsets.UTF_8));
		} catch (IOException e) {
			this.logProgress("Raw-file creation has thrown an exception: " + e.getMessage());
			e.printStackTrace();
		}
		
		
		HashSet<String> output = new HashSet<String>();



		
		StmtIterator iter = model.listStatements();
		int tmp = 0;
		while (iter.hasNext()) {
			Statement stmt = iter.next();
			tmp++;
		}
		int total = tmp;
		tmp = 0;
		iter = model.listStatements();
		try {
			while (iter.hasNext()) {
				if (tmp%10000==0) {
					if (indexed) {
						this.logProgress("\rIndexing: "+(int) ((double) tmp *100.0d/(double) total) + "%\r");
					} else {
						this.logProgress("\rPrefiltering: "+ (int) ((double) tmp *100.0d/(double) total) + "%\r");
					}
				}
				tmp++;
				
				Statement stmt = iter.next();

				Resource s = stmt.getSubject();
				Resource p = stmt.getPredicate();
				RDFNode o = stmt.getObject();
				
				if (!s.isURIResource()) {
					continue;
				}
				
				String text = "";
				if (o.isLiteral()) {
					String nid = s.getURI().toString();
					String lit = o.asLiteral().toString().replaceAll("[^A-Za-z0-9 ]", " ").replaceAll(" {2,}", " ")
							.replaceAll("^ ", "");
					text = "<" + nid + "> <" + p.getURI().toString() + "> \"" + lit + "\" .\n";
					if (p.getURI().toString().toLowerCase().equals("http://www.w3.org/2000/01/rdf-schema#label"))
						if (indexed) {
							blocker.addDoc(s.getURI().toString().toLowerCase(), lit);
							if (!cased_values.containsKey(s.getURI().toString().toLowerCase())) {
								cased_values.put(s.getURI().toLowerCase(), s.getURI());
							}
						} else if (!indexed) {
							
							boolean useMatchForGS = false;
							ArrayList<Tuple> result = blocker.searchFuzzyQuery(lit);
							for (int i = 0; i < result.size(); i++) {
								
								
								Tuple possibleMatch = result.get(i);
								String uri = possibleMatch.uri;
								uri = cased_values.get(uri);
								float score = possibleMatch.score;
								if (possibleMatch.score > 0.5) 
									writeFile(blockedwriter, uri + "\t" + nid + "\n");
								if (i == 0) {
									if (score > 1.5f && result.size() > 1) {
										if ((this.calculate(uri, nid) / Math.min(15,
																			 Math.max(uri.length(), nid.length()))) <
																			 0.05) {
										//if (result.get(i + 1).score / score < 0.5) {
											writeFile(gswriter, uri + "\t" + nid + "\t1\n");
											useMatchForGS = true;
										}
									}
								} else if (useMatchForGS) {
									writeFile(gswriter, uri + "\t" + nid + "\t0\n");
								}
							}
						}
				}
			}
			if (indexed) {
				this.logProgress("Indexing: 100%\n");
			} else {
				this.logProgress("Prefiltering: 100%\n");
			}
		} catch(Exception e) {
			this.logProgress("Indexing or pre-filtering has thrown an exception: " + e.getMessage());
			e.printStackTrace();
		}finally {
			if (iter != null)
				iter.close();
		}

		try {
			if (indexed)
				blocker.closeIndexing();
			blockedwriter.close();
			gswriter.close();
			// ntwriter.close();
		} catch (Exception e) {
			this.logProgress("Closing writers has thrown exception: " + e.getMessage());
			e.printStackTrace();
		}

	}

	private void writeFile(BufferedWriter writer, String text) {
		try {
			writer.write(text);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public int calculate(String x, String y) {
		x = x.substring(0, Math.min(x.length(), 15));
		y = y.substring(0, Math.min(y.length(), 15));
		int[][] dp = new int[x.length() + 1][y.length() + 1];

		for (int i = 0; i <= x.length(); i++) {
			for (int j = 0; j <= y.length(); j++) {
				if (i == 0) {
					dp[i][j] = j;
				} else if (j == 0) {
					dp[i][j] = i;
				} else {
					dp[i][j] = min(dp[i - 1][j - 1] + costOfSubstitution(x.charAt(i - 1), y.charAt(j - 1)),
							dp[i - 1][j] + 1, dp[i][j - 1] + 1);
				}
			}
		}

		return dp[x.length()][y.length()];
	}

	public int costOfSubstitution(char a, char b) {
		return a == b ? 0 : 1;
	}

	public int min(int... numbers) {
		return Arrays.stream(numbers).min().orElse(Integer.MAX_VALUE);
	}

	public class Blocker {

		public Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_40);
		public IndexWriterConfig config = new IndexWriterConfig(Version.LUCENE_40, analyzer);
		public RAMDirectory ramDirectory = new RAMDirectory();
		public IndexWriter indexwriter;
		public IndexReader idxReader = null;
		public IndexSearcher idxSearcher = null;

		public Blocker() {
			try {
				indexwriter = new IndexWriter(ramDirectory, config);

			} catch (Exception e) {
				System.err.print("Initializing index has thrown an exception: " + e.getMessage());
				e.printStackTrace();
			}
		}

		public void closeIndexing() {

			try {
				indexwriter.close();
				idxReader = DirectoryReader.open(ramDirectory);
				idxSearcher = new IndexSearcher(idxReader);
				// ramDirectory.close();
			} catch (IOException e) {
				System.err.print("Initializing index-tools has thrown an exception: " + e.getMessage());
				e.printStackTrace();
			}

		}

		public void createDoc(String uri, String label) throws IOException {
			Document doc = new Document();
			doc.add(new StringField("uri", uri, Field.Store.YES));
			doc.add(new TextField("label", label, Field.Store.YES));

			indexwriter.addDocument(doc);
		}

		public void addDoc(String uri, String label) {
			try {
				createDoc(uri, label);

			} catch (Exception e) {
				System.err.print("Lucene-doc-creation has thrown an exception: " + e.getMessage());
				e.printStackTrace();
			}
		}

		public class Tuple {
			String uri;
			float score;

			public Tuple(String uri, float score) {
				this.uri = uri;
				this.score = score;
			}
		}

		public ArrayList<Tuple> searchIndexAndDisplayResults(Query query) {
			ArrayList<Tuple> result = new ArrayList<>();
			try {

				TopDocs docs = idxSearcher.search(query, 3);
				String l1 = query.toString().replaceAll("~", " ");
				for (ScoreDoc doc : docs.scoreDocs) {
					Document thisDoc = idxSearcher.doc(doc.doc);
					String l2 = thisDoc.get("label");
					//if (doc.score > 0.8)// (calculate(l1, l2) / Math.min(15, Math.max(l1.length(), l2.length())) < 0.3)
					result.add(new Tuple(thisDoc.get("uri"), doc.score));
				}
			} catch (IOException e) {
				System.err.print("Index-search has thrown an exception: " + e.getMessage());
				e.printStackTrace();
			}
			return result;
		}

		public int calculate(String x, String y) { 
			x = x.substring(0, Math.min(x.length(), 15));
			y = y.substring(0, Math.min(y.length(), 15));
			int[][] dp = new int[x.length() + 1][y.length() + 1];

			for (int i = 0; i <= x.length(); i++) {
				for (int j = 0; j <= y.length(); j++) {
					if (i == 0) {
						dp[i][j] = j;
					} else if (j == 0) {
						dp[i][j] = i;
					} else {
						dp[i][j] = min(dp[i - 1][j - 1] + costOfSubstitution(x.charAt(i - 1), y.charAt(j - 1)),
								dp[i - 1][j] + 1, dp[i][j - 1] + 1);
					}
				}
			}

			return dp[x.length()][y.length()];
		}

		public int costOfSubstitution(char a, char b) {
			return a == b ? 0 : 1;
		}

		public int min(int... numbers) {
			return Arrays.stream(numbers).min().orElse(Integer.MAX_VALUE);
		}

		public ArrayList<Tuple> searchFuzzyQuery(String searchstring) {
			String[] splitstring = searchstring.replaceAll("[^A-Za-z0-9 ]", " ").replaceAll(" {2,}", " ")
					.replaceAll("^ ", "").toLowerCase().split(" ");
			Arrays.sort(splitstring, (a, b) -> Integer.compare(b.length(), a.length()));
			splitstring = Arrays.copyOfRange(splitstring, 0, 3);
			for (int i = 0; i < splitstring.length; i++) {
				splitstring[i] = splitstring[i] + "~";
			}
			searchstring = String.join(" ", splitstring);
			Query fuzzyQuery = null;
			try {
				// fuzzyQuery = new ComplexPhraseQueryParser(Version.LUCENE_40, "label",
				// analyzer).parse(searchstring);

				StandardAnalyzer analyzer = new StandardAnalyzer(Version.LUCENE_40);
				QueryParser parser = new QueryParser(Version.LUCENE_40, "label", analyzer);
				parser.setDefaultOperator(QueryParser.Operator.OR);
				fuzzyQuery = parser.parse(searchstring);

			} catch (ParseException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return searchIndexAndDisplayResults(fuzzyQuery);
		}

	}

}
