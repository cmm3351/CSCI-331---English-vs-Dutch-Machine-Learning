/**
 * This program contains functions to perform Decision
 * Tree or AdaBoost Learning algorithms on a given file
 * of English and Dutch sentences, either predicting
 * their exact languages based of characteristics or
 * creating a model to do so.
 *
 * Lab 3 for CSCI 331
 * @author Crisitan Malone, cmm3351
 */

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Object class used for AdaBoost training to store each
 * stump's question and its initial importance. They will
 * eventually be placed in a PriorityQueue and selected
 * based off highest importance.
 */
class Stump implements Comparable<Stump> {

    /** String question of this stump */
    String question;

    /** Double importance value of this stump */
    double importance;

    /**
     * Constructor for Stump class
     *
     * @param question String question of this stump
     * @param importance Double importance of this stump
     */
    public Stump(String question, double importance) {
        this.question = question;
        this.importance = importance;
    }

    /**
     * Getter for String question
     *
     * @return String question
     */
    public String getQuestion() {
        return question;
    }

    /**
     * Overridden compareTo function to ensure stump
     * objects will be ordered based on highest importance
     * in later PriorityQueue.
     *
     * @param other the object to be compared.
     * @return -1, 0, or 1 depending on if other.importance
     *          is greater than, equal to, or less than this.importance.
     */
    @Override
    public int compareTo(Stump other) {
        return Double.compare(other.importance, this.importance);
    }
}

/**
 * Class used to represent and store the generated models
 * for either Decision Tree or AdaBoost Learning.
 */
class Node implements Serializable {

    /** String question of this node */
    String question;

    /** Child nodes, or the next questions asked/
     *  results found, based on the answers of
     *  this node's question */
    HashMap<String, Node> childNodes = new HashMap<>();

    /** String result of this node, if it is a result node */
    String result;

    /** Boolean value indicating if this model represents
     *  a Decision Tree or AdaBoost model. False by default,
     *  therefore indicating a Decision Tree. */
    Boolean ada = false;

    /** If this is an AdaBoost model, this field points
     *  to the next stump to test */
    Node nextStump = null;

    /** If this is an AdaBoost model, this field indicates
     *  the hypothesis weight of this stump */
    Float adaHypWeight;

    /**
     * Constructor for Node class
     *
     * @param question String question being asked by this node. Null if result node.
     * @param result String result stored in this node. Null if question node.
     */
    public Node(String question, String result) {
        this.question = question;
        this.result = result;
    }

    /**
     * Getter for String question
     *
     * @return String question
     */
    public String getQuestion() {
        return question;
    }

    /**
     * Getter for HashMap<String,Node> childNodes
     *
     * @return HashMap<String,Node> childNodes
     */
    public HashMap<String,Node> getChildNodes() {
        return childNodes;
    }

    /**
     * Getter for String result
     *
     * @return String result
     */
    public String getResult() {
        return result;
    }

    /**
     * Getter for boolean ada
     *
     * @return boolean ada
     */
    public Boolean getAda() {
        return ada;
    }

    /**
     * Getter for Node nextStump
     *
     * @return Node nextStump
     */
    public Node getNextStump() {
        return nextStump;
    }

    /**
     * Getter for float adaHypWeight
     *
     * @return float adaHypWeight
     */
    public Float getAdaHypWeight() {
        return adaHypWeight;
    }

    /**
     * Setter for boolean ada
     *
     * @param ada Boolean ada to set
     */
    public void setAda(Boolean ada) {
        this.ada = ada;
    }

    /**
     * Getter for Node nextStump
     *
     * @param nextStump Bode nextStump to set
     */
    public void setNextStump(Node nextStump) {
        this.nextStump = nextStump;
    }

    /**
     * Setter for float adaHypWeight
     *
     * @param adaHypWeight float adaHypWeight to set
     */
    public void setAdaHypWeight(Float adaHypWeight) {
        this.adaHypWeight = adaHypWeight;
    }

}

/**
 * This is the main class for this program. It contains
 * several function to perform importance/entropy calculations,
 * decision tree learning, and AdaBoost algorithms on the
 * inputted language data to create and utilize predictive
 * models.
 */
public class lab3 {

    /**
     * ----------------------------
     * ENTROPY/IMPORTANCE FUNCTIONS
     * ----------------------------
     */

    /**
     * Helper function used for the calculation of the remainder
     * for gain, in the overall calculation of importance of the
     * current question being asked for the given sample of examples.
     *
     * @param aCount float count of all positive (en) examples
     * @param bCount float count of all positive (nl) examples
     * @param aTrueCount float count of all positive (en) examples that
     *                   have an answer of True for the current question
     * @param aFalseCount float count of all positive (en) examples that
     *                    have an answer of False for the current question
     * @param bTrueCount float count of all negative (nl) examples that
     *                   have an answer of True for the current question
     * @param bFalseCount float count of all negative (nl) examples that
     *                    have an answer of False for the current question
     * @return Double value representing the gain's remainder in the importance
     *         calculation
     */
    private static double remainder(float aCount, float bCount, float aTrueCount, float aFalseCount, float bTrueCount, float bFalseCount) {

        double remainder;

        remainder = ((aTrueCount+bTrueCount)/(aCount+bCount)) * booleanEntropy(aTrueCount/(aTrueCount+bTrueCount));

        remainder += ((aFalseCount+bFalseCount)/(aCount+bCount)) * booleanEntropy(aFalseCount/(aFalseCount+bFalseCount));

        return remainder;
    }

    /**
     * Helper function used to calculate the Boolean Entropy
     * of a given probability.
     *
     * @param prob Float probability for calculations
     * @return Double value representing the Boolean entropy of prob
     */
    private static double booleanEntropy(float prob) {

        if (prob == 0 || prob == 1 || Float.isNaN(prob)) {
            return 0;
        }

        double e1 = prob * (Math.log(1/prob) / Math.log(2));

        double e2 = (1-prob) * (Math.log(1/(1-prob)) / Math.log(2));

        return e1 + e2;
    }

    /**
     * This function calculates the importance of a given attr/question for
     * a given group of examples, indicating how good the question is at
     * partitioning the examples based on their result classification (language).
     * This is used to help pick the order or next question to ask in
     * both learning algorithms.
     *
     * @param attr String representing current question
     * @param examples ArrayList<HashMap<String, String>> representing examples
     *                 stored at the current Node.
     * @return Double value representing attr's importance.
     */
    private static double importance(String attr, ArrayList<HashMap<String, String>> examples) {

        double gain;

        float aCount = 0;
        float bCount = 0;
        float aTrueCount = 0;
        float aFalseCount = 0;
        float bTrueCount = 0;
        float bFalseCount = 0;

        for (HashMap<String, String> example : examples) {
            String resultAnswer = example.get("Result");
            String attrAnswer = example.get(attr);

            if (resultAnswer.equals("en")) {
                aCount++;

                if (attrAnswer.equals("True")) {
                    aTrueCount++;
                }
                else {
                    aFalseCount++;
                }

            } else if (resultAnswer.equals("nl")) {
                bCount++;

                if (attrAnswer.equals("True")) {
                    bTrueCount++;
                }
                else {
                    bFalseCount++;
                }
            }
        }

        gain = booleanEntropy(aCount/(aCount+bCount)) - remainder(aCount, bCount, aTrueCount, aFalseCount, bTrueCount, bFalseCount);

        return gain;
    }

    /**
     * --------------------------------
     * DECISION TREE LEARNING FUNCTIONS
     * --------------------------------
     */

    /**
     * Helper function for Decision Tree Learning algorithm. When a group of
     * examples can no longer be partitioned, it is used to generate a result
     * node by finding what the majority of the examples remaining are
     * classified as.
     *
     * @param examples ArrayList<HashMap<String, String>> examples to be analyzed
     * @return New result Node, indicating that the answer to the parent question
     *         results in an English (en) or Dutch (nl) classification.
     */
    private static Node majorityAnswer(ArrayList<HashMap<String, String>> examples) {


        int aCount = 0;
        int bCount = 0;

        for (HashMap<String, String> example : examples) {
            String answer = example.get("Result");

            if (answer.equals("en")) {
                aCount++;
            } else if (answer.equals("nl")) {
                bCount++;
            }
        }

        if (aCount >= bCount) {
            return new Node("Result", "en");
        }
        else {
            return new Node("Result", "nl");
        }

    }


    /**
     * Recursive function for generating a Decision Tree based on given examples,
     * attributes, and other information.
     *
     * @param examples ArrayList<HashMap<String, String>> examples containing
     *                 attributes to analyze
     * @param attributes ArrayList of String questions/attributes of each example
     * @param parentExamples ArrayList<HashMap<String, String>> examples from parent
     *                       recursion level
     * @param depth Integer representing current tree/recursive depth of this call.
     * @param maxDepth Integer representing the maximum depth the tree should reach
     *                 before it hase to generate result nodes. Used for AdaBoost.
     * @param stump String representing a question. It is used to force the algorithm
     *              generate children and find results based of this question instead
     *              of finding each question's importance specifically for AdaBoost.
     * @return Node containing the finished Decision Tree.
     */
    private static Node decisionTreeLearning(ArrayList<HashMap<String, String>> examples, ArrayList<String> attributes, ArrayList<HashMap<String, String>> parentExamples,
                                             int depth, int maxDepth, String stump) {

        int numExamples = examples.size();

        if (numExamples > 0) {
            String firstResult;
            boolean same = true;
            firstResult = examples.get(0).get("Result");

            for (int i = 1; i < numExamples; i++) {
                if (firstResult.equals(examples.get(i).get("Result"))) {
                    same = false;
                }
            }

            if (same) {
                return majorityAnswer(examples);
            }
        }

        if (attributes.isEmpty() || depth == maxDepth) {
            return majorityAnswer(examples);
        }
        else if (examples.isEmpty()) {
            return majorityAnswer(parentExamples);
        }

        String bestQuestion = null;
        double maxImportance = 0;
        int i = 0;
        Node newNode;

        if (stump == null) {
            for (String attr : attributes) {
                double currImportance = importance(attr, examples);

                if (i == 0 || currImportance > maxImportance) {
                    maxImportance = currImportance;
                    bestQuestion = attr;
                }

                i++;
            }

            newNode = new Node(bestQuestion, null);
        }
        else {
            bestQuestion = stump;

            newNode = new Node(bestQuestion, null);
            newNode.setAda(true);
        }

        ArrayList<HashMap<String, String>> trueExamples = new ArrayList<>();
        ArrayList<HashMap<String, String>> falseExamples = new ArrayList<>();


        for (HashMap<String,String> ex : examples) {
            String currAnswer = ex.get(bestQuestion);

            if (currAnswer.equals("True")) {
                trueExamples.add(ex);
            }
            else {
                falseExamples.add(ex);
            }

        }

        ArrayList<String> newTrueAttributes = new ArrayList<>(List.copyOf(attributes));
        ArrayList<String> newFalseAttributes = new ArrayList<>(List.copyOf(attributes));
        newTrueAttributes.remove(bestQuestion);
        newFalseAttributes.remove(bestQuestion);

        newNode.getChildNodes().put("True",decisionTreeLearning(trueExamples, newTrueAttributes, examples, depth + 1, maxDepth, stump));
        newNode.getChildNodes().put("False",decisionTreeLearning(falseExamples, newFalseAttributes, examples, depth + 1, maxDepth, stump));

        return newNode;
    }

    /**
     * ----------------------------
     * ADABOOST LEARNING FUNCTIONS
     * ----------------------------
     */

    /**
     * Helper function used in AdaBoost learning to determine
     * if an example was classified by a stump correctly during
     * training.
     *
     * @param ex HashMap<String, String> example being tested
     * @param model Node model containing current Decision stump
     * @return If the examples was classified correctly, the
     *         function will return true. Otherwise, it will
     *         return false.
     */
    private static boolean testExample(HashMap<String, String> ex, Node model) {

        Node currNode = model;

        while (currNode.getResult() == null) {
            String currQ = currNode.getQuestion();
            String exAnswer = ex.get(currQ);

            if (exAnswer.equals("True")) {
                currNode = currNode.getChildNodes().get("True");
            }
            else {
                currNode = currNode.getChildNodes().get("False");
            }
        }

        return ex.get("Result").equals(currNode.getResult());
    }

    /**
     * Helper function used by AdaBoost learning to normalize all
     * the modified weights of each example after error checking.
     * This ensures their weights sum to 1 before moving on to the
     * next hypothesis.
     *
     * @param examples ArrayList<HashMap<String, String>> examples to normalize
     */
    private static void normalizeWeights(ArrayList<HashMap<String, String>> examples) {

        float totalWeight = 0.0f;

        for (HashMap<String,String> ex : examples) {
            totalWeight += Float.parseFloat(ex.get("Weight"));
        }

        for (HashMap<String,String> ex : examples) {
            float newWeight = Float.parseFloat(ex.get("Weight")) / totalWeight;
            ex.put("Weight", String.valueOf(newWeight));
        }

    }

    /**
     * This is the main function for performing AdaBoost on a
     * given set of examples. It utilizes decisionTreeLearning()
     * as a base learning algorithm for generating decision stumps,
     * generating each stump in an order determined by the PriorityQueue
     * stumpQueue. It then reassigns new weights to each example
     * based on whether the stump correctly classifies it, normalizes
     * them, assigns a hypothesis weight to that stump based
     * on its error weight, gets the next stump, and repeats.
     *
     * @param examples ArrayList<HashMap<String, String>> weighted examples to test
     * @param attributes ArrayList<String> ArrayList of strings containing all questions
     * @param stumpQueue PriorityQueue<Stump> used to determine which stump to test next
     *                   based on calculated importance.
     * @return Node containing all the decision stumps and their weights, linked in
     *         the order they were tested.
     */
    private static Node adaBoost(ArrayList<HashMap<String, String>> examples, ArrayList<String> attributes, PriorityQueue<Stump> stumpQueue) {

        String currQuestion;
        Node firstHyp = null;
        Node parentHyp = null;

        while (!stumpQueue.isEmpty()) {

            currQuestion = stumpQueue.poll().getQuestion();

            Node currHyp = decisionTreeLearning(examples, attributes, null, 0, 1, currQuestion);
            float err = 0;

            for (HashMap<String,String> ex : examples) {
                boolean correct = testExample(ex, currHyp);

                if (!correct) {
                    err += Float.parseFloat(ex.get("Weight"));
                }
            }

            float deltaW = (err)/(1-err);

            for (HashMap<String,String> ex : examples) {
                boolean correct = testExample(ex, currHyp);

                if (correct) {
                    float newWeight = Float.parseFloat(ex.get("Weight")) * deltaW;
                    ex.put("Weight", String.valueOf(newWeight));
                }
            }

            normalizeWeights(examples);

            double hypWeight = 0.5 * Math.log((1-err)/err);
            currHyp.setAdaHypWeight((float) hypWeight);

            if (firstHyp == null) {
                firstHyp = currHyp;
            }
            else {
                parentHyp.setNextStump(currHyp);
            }
            parentHyp = currHyp;
        }

        return firstHyp;
    }

    /**
     * -------------------------
     * CLASSIFICATION FUNCTIONS
     * -------------------------
     */

    /**
     * This function is used to read the examples for training or
     * predicting from a file and classify them along each attribute,
     * storing the question answers in a HashMap.
     * For training, the function also assigns the result specified
     * for each example to each HashMap.
     * The function also assigns a Weight to each example of 1/N,
     * where N is the number of examples. This will only be used
     * by AdaBoost algorithms.
     *
     * @param filename String filename of file containing examples
     * @param training Boolean indicating if this call is for training
     *                 or predicting, which determines if it should
     *                 look for results in the file as well.
     * @return ArrayList<HashMap<String, String>> containing each example's
     *         answers for each question, their result if known, and their
     *         weight.
     */
    private static ArrayList<HashMap<String, String>> answerQuestions(String filename, boolean training) {

        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));

            String exampleStr = br.readLine();
            ArrayList<HashMap<String,String>> examples = new ArrayList<>();

            while (exampleStr != null) {
                String[] exStrArr = exampleStr.trim().split("\\s+");
                ArrayList<String> exStrList = new ArrayList<>(List.of(exStrArr));

                if (exStrList.size() < 15) {
                    System.out.println();
                }

                String word;
                int charCount = 0;
                HashMap<String, String> currEx = new HashMap<>();

                for (int i = 0; i < 15; i++) {

                    word = exStrList.get(i);

                    if (i == 0) {
                        if (training) {
                            String[] splitClass = word.split("\\|");
                            currEx.put("Result", splitClass[0]);
                            word = splitClass[1];
                        }
                        else {
                            currEx.put("Result", null);
                        }
                    }

                    word = word.replaceAll("[,.!?;:()]","").toLowerCase();

                    charCount += word.length();

                    if (word.contains("aa") && !currEx.containsKey("Q1")) {
                        currEx.put("Q1", "True");
                    }
                    else if (i == 14 && !currEx.containsKey("Q1")) {
                        currEx.put("Q1", "False");
                    }

                    if (word.contains("j") && !currEx.containsKey("Q2") && !word.contains("ja") && !word.contains("je") && !word.contains("ji")
                            && !word.contains("jo") && !word.contains("jo") && !word.contains("ju") && !word.contains("jy")) {
                        currEx.put("Q2", "True");
                    }
                    else if (i == 14 && !currEx.containsKey("Q2")) {
                        currEx.put("Q2", "False");
                    }

                    if (word.equals("de") && !currEx.containsKey("Q3")) {
                        currEx.put("Q3", "True");
                    }
                    else if (i == 14 && !currEx.containsKey("Q3")) {
                        currEx.put("Q3", "False");
                    }

                    if (word.equals("het") && !currEx.containsKey("Q4")) {
                        currEx.put("Q4", "True");
                    }
                    else if (i == 14 && !currEx.containsKey("Q4")) {
                        currEx.put("Q4", "False");
                    }

                    if (word.equals("the") && !currEx.containsKey("Q5")) {
                        currEx.put("Q5", "True");
                    }
                    else if (i == 14 && !currEx.containsKey("Q5")) {
                        currEx.put("Q5", "False");
                    }

                    if (word.equals("een") && !currEx.containsKey("Q6")) {
                        currEx.put("Q6", "True");
                    }
                    else if (i == 14 && !currEx.containsKey("Q6")) {
                        currEx.put("Q6", "False");
                    }

                    if (word.equals("a") && !currEx.containsKey("Q7")) {
                        currEx.put("Q7", "True");
                    }
                    else if (i == 14 && !currEx.containsKey("Q7")) {
                        currEx.put("Q7", "False");
                    }

                    if (word.equals("an") && !currEx.containsKey("Q8")) {
                        currEx.put("Q8", "True");
                    }
                    else if (i == 14 && !currEx.containsKey("Q8")) {
                        currEx.put("Q8", "False");
                    }
                }

                float avgCharCount = (float) charCount / 15;

                if (avgCharCount >= 5) {
                    currEx.put("Q0", "True");
                }
                else {
                    currEx.put("Q0", "False");
                }

                examples.add(currEx);

                exampleStr = br.readLine();
            }

            br.close();

            float adaWeight = 1.0f / examples.size();

            for (HashMap<String, String> ex : examples) {
                ex.put("Weight", String.valueOf(adaWeight));
            }

            return examples;

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * This function is used to determine whether each example in
     * a list of examples is either English (en) or Dutch (nl) based
     * on a given model. The function first checks to see if the
     * model is a Decision Tree or a set of weight hypotheses from
     * AdaBoost.
     * If the model is a decision tree, for each example, the function
     * will traverse through the tree based on the example's attributes
     * until it reaches a result. It will then print that result to the
     * console.
     * If the model is a set of weighted hypotheses, for each example,
     * the function will test that example across each decision stump
     * individually, assigning a 1 for (en) results and a -1 for (nl)
     * results. It will then multiply that value by the corresponding
     * stump's weight, and sum those calculations together for every
     * stump. If the sum is greater than or equal to 0, the algorithm
     * will classify it as English (en), and print that to the console.
     * Otherwise, it will classify it as Dutch (nl), and print that
     * to the console.
     *
     * @param examples ArrayList<HashMap<String, String>> examples to be
     *                 analyzed
     * @param model Node model used to perform predictions.
     */
    private static void predict(ArrayList<HashMap<String, String>> examples, Node model) {

        if (!model.getAda()) {

            for (HashMap<String, String> ex : examples) {

                Node currNode = model;

                while (currNode.getResult() == null) {
                    String currQ = currNode.getQuestion();
                    String exAnswer = ex.get(currQ);

                    if (exAnswer.equals("True")) {
                        currNode = currNode.getChildNodes().get("True");
                    } else {
                        currNode = currNode.getChildNodes().get("False");
                    }
                }

                if (currNode.getResult().equals("en")) {
                    System.out.println("en");
                } else {
                    System.out.println("nl");
                }

            }
        }
        else {

            for (HashMap<String, String> ex : examples) {

                Node currStump = model;
                float majority = 0.0f;

                while (currStump != null) {

                    String currQ = currStump.getQuestion();
                    float hypWeight = currStump.getAdaHypWeight();
                    String exAnswer = ex.get(currQ);
                    Node result;

                    if (exAnswer.equals("True")) {
                        result = currStump.getChildNodes().get("True");
                    } else {
                        result = currStump.getChildNodes().get("False");
                    }

                    if (result.getResult().equals("en")) {
                        majority += (1 * hypWeight);
                    } else {
                        majority += (-1 * hypWeight);
                    }

                    currStump = currStump.getNextStump();
                }

                if (majority >= 0) {
                    System.out.println("en");
                }
                else {
                    System.out.println("nl");
                }
            }
        }
    }

    /**
     * --------------
     * MAIN FUNCTION
     * --------------
     */

    /**
     * Main function for this program. It first parses through the command
     * line args, looking the first argument to determine if the user wants
     * to train or predict data.
     * If the user wants to train, it will call answerQuestions() to
     * generate the features of each example for training. It will then
     * determine which algorithm the user wants to use and call the
     * necessary functions to set up and start those processes. Once
     * completed, it will save the finished model's object to a
     * serialized file, named after another command line argument.
     * If the user wants to predict, it will call answerQuestions() to
     * generate the features of each example for predicting. It will
     * then deserialize an object file, specified by a command line
     * argument, containing the model to use. Then, it will call predict()
     * on the data and model, which will then handle determining the
     * type of model and printing the results to the console.
     *
     *
     * @param args String array of command line arguments.
     */
    public static void main(String[] args) {

        ArrayList<String> attributes = new ArrayList<>();
        attributes.add("Q0");
        attributes.add("Q1");
        attributes.add("Q2");
        attributes.add("Q3");
        attributes.add("Q4");
        attributes.add("Q5");
        attributes.add("Q6");
        attributes.add("Q7");
        attributes.add("Q8");

        String action = args[0];

        if (action.equals("train")) {
            String examples = args[1];
            String hypothesisOut = args[2];
            String learningType = args[3];

            ArrayList<HashMap<String, String>> dataList = answerQuestions(examples, true);

            if (learningType.equals("dt")) {
                Node decisionTree = decisionTreeLearning(dataList, attributes, null, 0, Integer.MAX_VALUE, null);

                try {
                    ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(hypothesisOut));
                    out.writeObject(decisionTree);
                    out.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

            }
            else if (learningType.equals("ada")) {

                PriorityQueue<Stump> stumpQueue = new PriorityQueue<>();

                for (String attr : attributes) {
                    double currImportance = importance(attr, dataList);

                    Stump stump = new Stump(attr, currImportance);
                    stumpQueue.add(stump);
                }

                Node weightedHyp = adaBoost(dataList, attributes, stumpQueue);

                try {
                    ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(hypothesisOut));
                    out.writeObject(weightedHyp);
                    out.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            else {
                System.out.println("Invalid Learning Type");
                System.exit(-1);
            }
        }
        else if (action.equals("predict")) {
            String hypothesis = args[1];
            String file = args[2];

            ArrayList<HashMap<String, String>> dataList = answerQuestions(file, false);

            try {
                ObjectInputStream in = new ObjectInputStream(new FileInputStream(hypothesis));
                Node model = (Node) in.readObject();
                in.close();

                predict(dataList, model);
            } catch (ClassNotFoundException | IOException e) {
                throw new RuntimeException(e);
            }

        }
        else {
            System.out.println("Invalid option.");
            System.exit(-1);
        }

    }
}