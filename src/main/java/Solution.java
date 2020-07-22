

import java.io.*;
import java.nio.charset.StandardCharsets;

/**
 * created with IntelliJ IDEA 2019.3
 * author: xig
 * date: 2020/6/6 9:31
 * version: 1.0
 * description: to do
 */
public class Solution {

    static String PATH = "F:\\corrected";
    static String OUT = "F:\\corrected.out";

    public static void main(String[] args) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(PATH), StandardCharsets.UTF_8));
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(OUT), StandardCharsets.UTF_8))) {

            String line;
            String[] fields;
            StringBuilder builder = new StringBuilder();

            while ((line = reader.readLine()) != null) {
                fields = line.split(",");
                if (fields.length != 42) {
                    continue;
                }
                String[] out = new String[fields.length - 1];
                System.arraycopy(fields, 0, out, 0, out.length);
                builder.append(String.join(",", out)).append("\n");
            }
            writer.write(builder.toString());


        } catch (Exception e) {
            e.printStackTrace();
        }


    }

}
