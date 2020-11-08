package mis_3c_quiz1;
public class Student {
    private String name;
    private int score1,score2,score3;
    private double avg;
    
    public Student(String n,int s1,int s2,int s3){
        this.name=n;
        this.score1=s1;
        this.score2=s2;
        this.score3=s3;
    }
    public void CalAverage(){
        avg=(score1+score2+score3)/3;
    }
    public void PrintResult(){
        System.out.printf("學生 %s\n分數一=%d\n分數二=%d\n分數三=%d\n平均分數=%.2f\n",name,score1,score2,score3,avg);
        if(avg>=60){
            System.out.println("Pass");
        }else{
            System.out.println("Fail");
        }
    }
}
