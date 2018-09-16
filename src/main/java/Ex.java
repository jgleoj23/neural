import java.util.ArrayList;
import java.util.List;

/**
 * @author Joseph Gardi
 */
public class Ex {

    public static void main(String[] args) {
        List<String> a = new ArrayList<>();
        a.add("hi");
        System.out.println("what");
        a.stream().map(element -> {
            System.out.println("hey");
            return 3;
        });
    }
}
