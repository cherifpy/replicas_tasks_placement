import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.chocosolver.solver.Model;
import org.chocosolver.solver.Settings;
import org.chocosolver.solver.Solution;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.constraints.nary.nvalue.amnv.rules.R;
import org.chocosolver.solver.exception.ContradictionException;
import org.chocosolver.solver.search.limits.FailCounter;
import org.chocosolver.solver.search.limits.ICounter;
import org.chocosolver.solver.search.loop.lns.neighbors.AdaptiveNeighborhood;
import org.chocosolver.solver.search.loop.lns.neighbors.INeighbor;
import org.chocosolver.solver.search.loop.lns.neighbors.SequenceNeighborhood;
import org.chocosolver.solver.search.loop.move.Move;
import org.chocosolver.solver.search.loop.move.MoveLNS;
import org.chocosolver.solver.search.restart.GeometricalCutoff;
import org.chocosolver.solver.search.strategy.Search;
import org.chocosolver.solver.search.strategy.decision.Decision;
import org.chocosolver.solver.search.strategy.decision.IntDecision;
import org.chocosolver.solver.search.strategy.selectors.values.*;
import org.chocosolver.solver.search.strategy.selectors.variables.*;
import org.chocosolver.solver.search.strategy.strategy.AbstractStrategy;
import org.chocosolver.solver.search.strategy.strategy.IntStrategy;
import org.chocosolver.solver.variables.BoolVar;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Task;
import org.chocosolver.solver.variables.Variable;
import org.chocosolver.util.sort.ArraySort;
import org.chocosolver.util.tools.ArrayUtils;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Offline COP model from the paper (Section 4.1): Choco-solver + Large Neighborhood
 * Search (LNS), objective = sum of flow times (objectives[0]) / max flow time
 * (objectives[1]).
 *
 * Copied from Flowtime 2/src/SchedulingWithDiffN.java (separate Java project), adapted
 * for this artifact: runScheduler now takes an explicit `outputDir` instead of a
 * hardcoded absolute path, and SchedulingResult now exposes the metrics (sumFlowTime,
 * maxFlowTime, nbTransfers, solveTimeSec, provenOptimal...) instead of only printing
 * them. Modeling/search logic unchanged.
 */
public class
SchedulingWithDiffN {

    // Enables/disables the bandwidth-sharing constraint (capacity=1 per node for
    // transfers). Enabled by default: the master has infinite outgoing bandwidth, only
    // the node's incoming bandwidth limits the transfer -- a node can therefore only
    // receive one transfer at a time. Can be disabled for experimentation.
    public static boolean ENABLE_BANDWIDTH_SHARING_CONSTRAINT = true;

    public static class state {
        int maxFlowTime;
        int timespane;

        public state(int maxFlowTime, int endTime) {
            this.maxFlowTime = maxFlowTime;
            this.timespane = endTime;
        }
    }

    public static class Job {
        IntVar startTime;
        IntVar endTime;

        public Job(IntVar startTime, IntVar endTime) {
            this.startTime = startTime;
            this.endTime = endTime;
        }
    }

    public static class TransferConfig {
        int jobIndex;
        int startTime;
        int endTime;
        int nodeIndex;

        public TransferConfig(int jobIndex, int startTime, int endTime, int nodeIndex) {
            this.jobIndex = jobIndex;
            this.startTime = startTime;
            this.endTime = endTime;
            this.nodeIndex = nodeIndex;
        }

    }

    public static class WorkConfig {
        int taskindex;
        int jobIndex;
        int startTime;
        int endTime;
        int nodeIndex;

        public WorkConfig(int taskindex, int jobindex, int startTime, int endTime, int nodeIndex) {
            this.taskindex = taskindex;
            this.jobIndex = jobindex;
            this.startTime = startTime;
            this.endTime = endTime;
            this.nodeIndex = nodeIndex;
        }
    }

    public static class SchedulingResult {
        public List<TransferConfig> transfers = new ArrayList<>();
        public List<WorkConfig> worksExec = new ArrayList<>();
        // Metrics of the best solution found (may be sub-optimal if the solver hit the
        // time limit before proving optimality - see foundSolution/provenOptimal).
        // Added for this artifact: the original code only printed these to stdout (see
        // onSolution), without exposing them to the caller.
        public boolean foundSolution = false;
        public boolean provenOptimal = false;
        public long sumFlowTime = -1;
        public long maxFlowTime = -1;
        public int nbTransfers = -1;
        public double solveTimeSec = -1;
        public long nbSolutionsFound = 0;

        public SchedulingResult(List<TransferConfig> transfers, List<WorkConfig> worksExec) {
            this.transfers = transfers;
            this.worksExec = worksExec;
        }
    }

    public static void writeWorkConfigCSV(List<WorkConfig> works, String path) throws Exception {
        FileWriter writer = new FileWriter(path);
        // Header
        writer.write("task_index,job_index,start_time,end_time,node_index\n");
        // Rows
        for (WorkConfig w : works) {
            writer.write(
            w.taskindex + "," +
                    w.jobIndex + "," +
                    w.startTime + "," +
                    w.endTime + "," +
                    w.nodeIndex + "\n"
            );
        }
        writer.close();
    }

    public static void writeTransferConfigCSV(List<TransferConfig> transfers, String path) throws Exception {
        FileWriter writer = new FileWriter(path);

        // header
        writer.write("job_index,start_time,end_time,node_index\n");

        // rows
        for (TransferConfig t : transfers) {
            writer.write(t.jobIndex + "," + t.startTime + "," + t.endTime + "," + t.nodeIndex + "\n");
        }

        writer.close();
    }

    /**
     * `outputDir` (can be null): directory to write works_exec_solution.csv/
     * transfers_solution.csv to on every improved solution found - replaces the
     * hardcoded absolute basePath from the original version (machine-specific), so this
     * artifact stays self-contained. null = write nothing to disk (the returned
     * SchedulingResult already has everything: transfers/worksExec/sumFlowTime/maxFlowTime/...).
     */
    public static SchedulingResult runScheduler(int nb_nodes, int nb_data, int[] data_sizes, int[][] works, int[] jobs_type, int[] bandwidths, double[] cpus, int[] storage_capacity, int[] starting_times, Model[] models, int pos, boolean solve, String outputDir) {
        final int CPU_UNIT = 1; // to scale cpu speeds
        // compute an upper bound on makespan (same idea as python)
        long makespanLong = 0;

        long sumData = 0;
        for (int s : data_sizes) sumData += s;

        int minBandwidth = Integer.MAX_VALUE;
        for (int b : bandwidths) if (b < minBandwidth) minBandwidth = b;

        makespanLong = sumData / Math.max(1, minBandwidth);

        long totalWork = 0;
        for (int[] wl : works) for (int w : wl) totalWork += w;

        double maxCpu = 0;
        for (double c : cpus) if (c > maxCpu) maxCpu = c;

        int maxStartingTime = 0;
        for (int s : starting_times) if (s > maxStartingTime) maxStartingTime = s;

        makespanLong += 0;
        makespanLong += totalWork * CPU_UNIT * Math.max(1, maxCpu);
        makespanLong *= 2;
        // The bound above only covers transfer+processing time from t=0; jobs
        // arriving late (starting_times) need that budget on top of their own
        // arrival, otherwise a slow-node transfer for a late job can end after
        // the bound, producing an invalid (lb > ub) IntVar domain at creation.
        makespanLong += maxStartingTime;

        int makespan = (int) Math.min(makespanLong, Integer.MAX_VALUE / 2);

        System.out.println("Computed makespan upper bound: " + makespan);
        // print inputs (summary)
        System.out.println("DATA");
        for (int i = 0; i < nb_data; i++) {
            System.out.println(" Data " + i + ": size=" + data_sizes[i] + " MB, works=" + Arrays.toString(works[i]) + " arrival=" + starting_times[i]);
        }
        System.out.println("NODES");
        for (int j = 0; j < nb_nodes; j++) {
            System.out.println(" Node " + j + ": bandwidth=" + bandwidths[j] + " MB/s, cpu=" + cpus[j] + " units/s");
        }

        System.out.printf("makespan = %d;%n", makespan);
        System.out.printf("nb_data = %d;%n", nb_data);
        System.out.printf("nb_nodes = %d;%n", nb_nodes);

        System.out.printf("data_sizes = %s;%n", Arrays.toString(data_sizes));
        System.out.printf("starting_times = %s;%n", Arrays.toString(starting_times));
        System.out.printf("nb_works = %s;%n", Arrays.toString(Arrays.stream(works).mapToInt(arr -> arr.length).toArray()));
        System.out.printf("work_duration = %s;%n", Arrays.toString(Arrays.stream(works).map(w -> w[0]).toArray()));
        System.out.printf("bandwidths = %s;%n", Arrays.toString(bandwidths));
        System.out.printf("cpus = %s;%n", Arrays.toString(cpus));

        // ----- MODEL -----
        Model model = new Model("Bag of Tasks Scheduling (Java)", Settings.dev().setLCG(false).setWarnUser(true));
        //Settings.dev()
        //        .setLCG(false)
        //        .setWarnUser(true);
        // Arrays for transfer tasks and heights
        Task[][] transferTasks = new Task[nb_nodes][nb_data];
        BoolVar[][] transferHeights = new BoolVar[nb_nodes][nb_data];

        // Create transfer tasks: one per (node, data)
        for (int j = 0; j < nb_nodes; j++) {
            for (int i = 0; i < nb_data; i++) {
                IntVar s = model.intVar("start_transfer_d" + i + "_n" + j, starting_times[i], makespan, true);
                int d = (int) Math.ceil((double) data_sizes[i] / (double) bandwidths[j]);
                IntVar durationVar = model.intVar(d);
                IntVar end = model.intVar("end_transfer_d" + i + "_n" + j, starting_times[i] + d, makespan, true);
                BoolVar h;
                if (data_sizes[i] > storage_capacity[j]) {
                    h = model.boolVar("height_transfer_d" + i + "_n" + j, false);
                    //model.arithm(h, "=", 0).post();
                }else{
                    h = model.boolVar("height_transfer_d" + i + "_n" + j);
                }

                Task t = new Task(s, durationVar, end);
                transferTasks[j][i] = t;
                transferHeights[j][i] = h;
                // Search-space reduction: if data i cannot fit in node j's storage
                // space, the transfer to j is impossible -> fix the variable to 0.

            }
        }

        // Create work tasks: for each node, each data, each wor
        //
        IntVar[][] jobStarts = new IntVar[nb_data][];
        IntVar[][] jobDurations = new IntVar[nb_data][];
        IntVar[][] jobEnds = new IntVar[nb_data][];
        IntVar[][] jobNodes = new IntVar[nb_data][];
        for (int i = 0; i < nb_data; i++) {
            int[] wl = works[i];
            jobStarts[i] = new IntVar[wl.length];
            jobDurations[i] = new IntVar[wl.length];
            jobEnds[i] = new IntVar[wl.length];
            jobNodes[i] = new IntVar[wl.length];

            // Search-space reduction: a node whose storage capacity is too small for
            // data i can never receive its transfer anyway (transferHeights fixed to 0
            // above), so no task tied to this data can run there either. Remove these
            // nodes directly from jobNodes' domain.
            List<Integer> validNodesList = new ArrayList<>();
            for (int j = 0; j < nb_nodes; j++) {
                if (data_sizes[i] <= storage_capacity[j]) validNodesList.add(j);
            }
            int[] validNodes = validNodesList.isEmpty()
                    ? ArrayUtils.array(0, nb_nodes - 1) // instance infaisable ; on laisse les autres contraintes le detecter
                    : validNodesList.stream().mapToInt(Integer::intValue).toArray();

            for (int k = 0; k < wl.length; k++) {
                int w = wl[k];
                jobStarts[i][k] = model.intVar("start_work_d" + i + "_w" + k, starting_times[i], makespan, true);
                int[] durations = new int[nb_nodes];
                for (int j = 0; j < nb_nodes; j++) {
                    durations[j] = (int) (w * cpus[j]);
                }
                int min = Arrays.stream(durations).min().getAsInt();
                int max = Arrays.stream(durations).max().getAsInt();
                jobDurations[i][k] = model.intVar("duration_work_d" + i + "_w" + k, min, max);
                jobNodes[i][k] = model.intVar("node_work_d" + i + "_w" + k, validNodes);
                model.element(jobDurations[i][k], durations, jobNodes[i][k]).post();
                jobEnds[i][k] = model.intVar("end_work_d" + i + "_w" + k, starting_times[i], makespan, true);
                model.arithm(jobStarts[i][k], "+", jobDurations[i][k], "=", jobEnds[i][k]).post();
                for (int j = 0; j < nb_nodes; j++) {
                    BoolVar jOnN = jobNodes[i][k].eq(j).boolVar();
                    // A work can start only after the corresponding transfer is finished on that node,
                    model.impXrelYC(jobStarts[i][k], ">=", transferTasks[j][i].getEnd(), 0, jOnN);
                    // and only if the transfer happened (height) -- managed by next set of constraints
                    //model.impXrelC(jOnN, "=", 0, transferHeights[j][i].not());
                    //jOnN.le(transferHeights[j][i]).post();
                }
            }
        }


        IntVar[][] nb_transfers = new IntVar[nb_data][nb_nodes];
        for (int i = 0; i < nb_data; i++) {
            for (int j = 0; j < nb_nodes; j++) {
                nb_transfers[i][j] = transferHeights[j][i].intVar();
            }
        }

        int factor = 1;
        for (int i = 0; i < nb_data; i++) {
            int[] wl = works[i];
            IntVar[] counters = new IntVar[nb_nodes];
            for (int j = 0; j < nb_nodes; j++) {
                counters[j] = model.intVar(0, wl.length);
                //model.count(j, jobNodes[i], counters[j]).post();
                model.reifXrelC(counters[j], ">=", 1, transferHeights[j][i]);
                // constraintes redondantes
                for (int k = 0; k < wl.length - 1; k++) {
                    transferHeights[j][i].eq(0).imp(jobNodes[i][k].ne(j)).post();
                }
            }

            model.globalCardinality(jobNodes[i], ArrayUtils.array(0, nb_nodes - 1), counters, true).post();
            model.sum(counters, "=", wl.length).post();

            // Transfer_time <= factor * sum(execution_time)
            for (int j = 0; j < nb_nodes && factor > 0; j++) {
                IntVar[] executions = new IntVar[works[i].length];
                for (int k = 0; k < executions.length; k++) {
                    executions[k] = model.isEq(jobNodes[i][k], j).mul(jobDurations[i][k]).intVar();
                }
                IntVar transfers_counter = model.intVar(0, nb_nodes);
                model.sum(nb_transfers[i], "=", transfers_counter).post();
                BoolVar h = transfers_counter.gt(1).and(transferHeights[j][i]).boolVar();
                model.sum(executions, ">=", model.intView(transferTasks[j][i].getDuration().getValue() * factor, h, 0)).post();
                //counters[j].gt(1).imp(exe).post();
            }

            for (int k = 0; k < wl.length - 1; k++) {
                jobNodes[i][k].eq(jobNodes[i][k + 1]).imp(jobEnds[i][k].eq(jobStarts[i][k + 1])).post();
                // very strict :
                jobNodes[i][k].le(jobNodes[i][k + 1]).post();
            }
        }

        ////  if a transfer happens, then at least one work must happen on that node for that data
        //for (int i = 0; i < nb_data; i++) {
        //    int[] wl = works[i];
        //    IntVar[] counters = new IntVar[nb_nodes];
        //    for (int j = 0; j < nb_nodes; j++) {
        //        counters[j] = model.intVar(0, wl.length);
        //        //model.count(j, jobNodes[i], counters[j]).post();
        //        model.reifXrelC(counters[j], ">=", 1, transferHeights[j][i]);
        //        // constraintes redondantes
        //        for (int k = 0; k < wl.length - 1; k++) {
        //            transferHeights[j][i].eq(0).imp(jobNodes[i][k].ne(j)).post();
        //        }
        //    }
        //    model.globalCardinality(jobNodes[i], ArrayUtils.array(0, nb_nodes - 1), counters, true).post();
        //    model.sum(counters, "=", wl.length).post();
        //    for (int k = 0; k < wl.length - 1; k++) {
        //        jobNodes[i][k].eq(jobNodes[i][k + 1]).imp(jobEnds[i][k].eq(jobStarts[i][k + 1])).post();
        //        // very strict :
        //        jobNodes[i][k].le(jobNodes[i][k + 1]).post();
        //    }
        //}


        // ----- STORAGE CONSTRAINT -----
        // Data i occupies storage on node j from the moment its transfer to j
        // starts until the last work assigned to that node for that data
        // finishes (release time) -- that's when it can be deleted locally.
        Task[][] storageTasks = new Task[nb_nodes][nb_data];
        IntVar[][] storageHeights = new IntVar[nb_nodes][nb_data];
        for (int i = 0; i < nb_data; i++) {
            int[] wl = works[i];
            for (int j = 0; j < nb_nodes; j++) {
                IntVar transferStart = transferTasks[j][i].getStart();

                // release_ij = max end time among works of data i actually assigned to node j;
                // falls back to transferStart when no work of i is on j (storage then irrelevant
                // since storageHeights[j][i] will be 0).
                IntVar[] candidateEnds = new IntVar[wl.length];
                for (int k = 0; k < wl.length; k++) {
                    BoolVar onJ = jobNodes[i][k].eq(j).boolVar();
                    IntVar cand = model.intVar("release_cand_d" + i + "_n" + j + "_w" + k, starting_times[i], makespan, true);
                    model.impXrelYC(cand, "=", jobEnds[i][k], 0, onJ);
                    model.impXrelYC(cand, "=", transferStart, 0, onJ.not());
                    candidateEnds[k] = cand;
                }
                IntVar release = model.intVar("release_d" + i + "_n" + j, starting_times[i], makespan, true);
                model.max(release, candidateEnds).post();

                IntVar storageDuration = model.intVar("storage_duration_d" + i + "_n" + j, 0, makespan, true);
                storageTasks[j][i] = new Task(transferStart, storageDuration, release);
                storageHeights[j][i] = transferHeights[j][i].mul(data_sizes[i]).intVar();
            }
        }
        for (int j = 0; j < nb_nodes; j++) {
            model.cumulative(storageTasks[j], storageHeights[j], model.intVar(storage_capacity[j])).post();
        }

        // ----- CONSTRAINTS -----
        // Cumulative constraints for transfers on each node (capacity = 1)
        // -> models bandwidth sharing (only one transfer at a time per node)
        if (ENABLE_BANDWIDTH_SHARING_CONSTRAINT) {
            for (int j = 0; j < nb_nodes; j++) {
                Task[] tasksForNode = new Task[nb_data];
                IntVar[] heightsForNode = new IntVar[nb_data];
                for (int i = 0; i < nb_data; i++) {
                    tasksForNode[i] = transferTasks[j][i];
                    heightsForNode[i] = transferHeights[j][i];
                }
                // capacity = 1
                // System.out.printf("Task_%d -- duration= %s%n", j, tasksForNode[0].getDuration());
                model.cumulative(tasksForNode, heightsForNode, model.intVar(1)).post();
            }
        }
        // At least one transfer per data (sum over nodes heights[j][i] >= 1)
        for (int i = 0; i < nb_data; i++) {
            IntVar[] arr = new IntVar[nb_nodes];
            for (int j = 0; j < nb_nodes; j++) arr[j] = transferHeights[j][i];
            model.sum(arr, ">=", 1).post();
        }

        // Each work must be done exactly once (sum of heights for a given (data i, work k) across nodes == 1)
        model.diffN(ArrayUtils.flatten(jobStarts),
                ArrayUtils.flatten(jobNodes),
                ArrayUtils.flatten(jobDurations),
                Arrays.stream(ArrayUtils.flatten(jobNodes)).map(j -> model.intVar(1)).toArray(IntVar[]::new),
                true).post();


        // Transfer_time <= factor * sum(execution_time)
        //int factor = 1;
        //for (int i = 0; i < nb_data && factor > 0; i++) {
        //    for (int j = 0; j < nb_nodes; j++) {
        //        IntVar[] executions = new IntVar[works[i].length];
        //        for (int k = 0; k < executions.length; k++) {
        //            IntVar job_duration = model.mul(jobDurations[i][k], (int)cpus[j]);
        //            executions[k] = model.isEq(jobNodes[i][k], j).mul(job_duration).intVar();
        //        }
        //        model.sum(executions, ">=", transferHeights[j][i], transferTasks[j][i].getDuration().getValue() * factor).post();
        //    }
        //}
        //IntVar[] nb_transfers = new IntVar[nb_data];
        //for (int i = 0; i < nb_data && factor > 0; i++) {
        //    IntVar[] tmp = new IntVar[nb_nodes];
        //    for (int j = 0; j < nb_nodes; j++) {
        //        tmp[j] = transferHeights[j][i];
        //    }
        //    model.sum(tmp, "=", nb_transfers[i]);
        //}

        //for (int i = 0; i < nb_data && factor > 0; i++) {
        //
        //    for (int j = 0; j < nb_nodes; j++) {
        //        IntVar[] executions = new IntVar[works[i].length];
        //        for (int k = 0; k < executions.length; k++) {
        //            IntVar job_duration = model.mul(jobDurations[i][k], (int)cpus[j]);
        //            executions[k] = model.isEq(jobNodes[i][k], j).mul(job_duration).intVar();
        //        }
        //        //IntVar par1 = model.mul()
        //        model.sum(executions, ">=", transferHeights[j][i], (int) Math.ceil((double) data_sizes[i] / (double) bandwidths[j]) * factor).post();
        //    }
        //
        //
        //}
        // ----- OBJECTIVE Make span-----
        //makespan var and ensure it's >= all end
        boolean makespan_obj = false;
        final IntVar[] objectives = new IntVar[3];
        //if (makespan_obj) {
            /*IntVar makespanVar = model.intVar("makespan", 0, makespan);

            // collect all work ends
            IntVar[] endsArr = new IntVar[workTasks.size()];
            int index = 0;
            for (Main.WorkEntry we : workTasks) {
                endsArr[index] = model.intVar(0, we.task.getEnd().getUB());
                model.impXrelYC(we.task.getEnd(), "=", endsArr[index], 0, we.height.asBoolVar());
                model.impXrelC(endsArr[index], "=", 0, we.height.asBoolVar().not());
                index++;
            }
            //
            new Constraint("max", new PropMax(endsArr, makespanVar)).post();
            //model.max(makespanVar, endsArr).post();// -- post max constraint between makespanVar and all ends
            // minimize makespan
            model.setObjective(false, makespanVar); // false => MINIMIZE (see Choco API)*/
        //} else {
        IntVar[] all_flow_time = new IntVar[nb_data];
        IntVar[] jobs_ends = new IntVar[nb_data];
        for (int i = 0; i < nb_data; i++) {
            jobs_ends[i] = model.intVar(0, makespan);
            model.max(jobs_ends[i], jobEnds[i]).post();

            //IntVar flow = model.intVar(0, makespan);
            //model.arithm(end_time, "-", flow, "=", starting_times[i]).post();
            IntVar flow = model.intView(1, jobs_ends[i], -starting_times[i]);
            all_flow_time[i] = flow;
        }
        IntVar maxFlowTime = model.intVar("max_flow_time", 0, 999_999);
        model.max(maxFlowTime, all_flow_time).post();

        IntVar sumFlowTime = model.intVar("sum_flow_time", 0, 999_999);
        model.sum(all_flow_time, "=", sumFlowTime).post();

        IntVar max_makespan = model.intVar("makespan", 0,999_999);
        model.max(max_makespan, jobs_ends).post();
        //model.setObjective(false, maxFlowTime);
        objectives[0] = sumFlowTime;
        objectives[1] = maxFlowTime;
        objectives[2] = max_makespan;
        //}

        //----- SOLVER -----
        Solver solver = model.getSolver();
        List<TransferConfig> transfersList = new ArrayList<>();
        List<WorkConfig> worksList = new ArrayList<>();
        model.displayVariableOccurrences();
        model.displayPropagatorOccurrences();

        IntVar[] decisionVars = decisionVariables(nb_nodes, nb_data, works, jobNodes, jobStarts, transferHeights, transferTasks);
        hints(nb_nodes, nb_data, data_sizes, works, cpus,  solver, jobNodes);//bandwidths,

        //solver.setNoGoodRecordingFromRestarts();
        ArraySort<?> sorter = new ArraySort<>(nb_nodes, false, true);
        int[] cidx = ArrayUtils.array(0, nb_nodes - 1);
        sorter.sort(cidx, nb_nodes, (i, j) -> (int) ((cpus[i] - cpus[j]) * 1000));
        /*final IntStrategy strat = Search.intVarSearch(
                new InputOrder<>(model), new IntDomainLast(model.getSolver().defaultSolution(), var -> {
            if (var.getName().startsWith("node_work_d")) {
                int i = 0;
                while (i < nb_nodes) {
                    if (var.contains(cidx[i])) {
                        return cidx[i];
                    }
                    i++;
                }
            }
            return var.getLB();
        }, (i, j) -> true), decisionVars);
        solver.setSearch(Search.lastConflict(new AbstractStrategy<>(model, decisionVars) {


            @Override
            public Decision<IntVar> getDecision() {
                IntDecision dec = (IntDecision) strat.getDecision();
                if (dec != null && dec.getDecisionVariable().getName().startsWith("start_")) {
                    dec.setRefutable(false);
                }
                return dec;
            }
        }, 2));
        */
        solver.setSearch(
                Search.lastConflict(
                        Search.intVarSearch(new InputOrder<>(model),
                                new IntDomainLast(model.getSolver().defaultSolution(),
                                        new IntDomainMin(), (i, j) -> true),
                                decisionVars), 2)
        );
        if (solve || pos == 1) {
            setLNS(solver,
                //new AdaptiveNeighborhood(42,
                new SequenceNeighborhood(
                    getNeighbor1(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds),
                    getNeighbor2(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds),
                    getNeighbor3(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds),
                    getNeighbor1bis(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds),
                    getNeighbor2bis(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds),
                    getNeighbor3bis(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds),
                    getNeighbor4bis(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds)
                ),
                new FailCounter(model, nb_data * nb_nodes * 100));
        } else if (pos == 2) {
            setLNS(solver,
                new SequenceNeighborhood(
                    getNeighbor1bis(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds),
                    getNeighbor3bis(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds)
                ),
                new FailCounter(model, nb_data * nb_nodes * 50));
        } else if (pos == 3) {
            setLNS(solver,
                new SequenceNeighborhood(
                    getNeighbor1(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds),
                    getNeighbor2bis(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds),
                    getNeighbor3bis(nb_data, works, starting_times, jobNodes, jobStarts, jobEnds)
                ),
                new FailCounter(model, nb_data * nb_nodes * 100));
        }
        //solver.setRestarts(i -> solver.getFailCount() > i, new GeometricalCutoff(600, 1.01), 50_000);

        solver.showShortStatistics();
        // OFFLINE_COP_TIME_LIMIT (environment variable, Choco format: "150s", "5m"...):
        // time budget per model - the CP problem is NP-hard, this limit is what makes
        // larger instances practical (the last solution found before expiry is
        // returned; see isObjectiveOptimal below to know whether it was also proven
        // optimal or just the best one found within the time budget).
        solver.limitTime(System.getenv("OFFLINE_COP_TIME_LIMIT") != null ? System.getenv("OFFLINE_COP_TIME_LIMIT") : "150s");
        //solver.showDecisions(() -> "" + model.getEnvironment().getWorldIndex());
        //solver.showContradiction();
        //solver.showRestarts(()->"R:"+model.getEnvironment().getWorldIndex());
        SchedulingResult result = new SchedulingResult(transfersList, worksList);
        solver.onSolution(() -> {
            result.foundSolution = true;
            //if (makespan_obj) {
            //    System.out.println("Solution found with max makespan = " + model.getObjective().asIntVar().getValue());
            //} else {
            //    System.out.println("Solution found with max Flow Time = " + model.getObjective().asIntVar().getValue());
            //}

            transfersList.clear();
            worksList.clear();
            boolean print = false;
            for (int j = 0; j < nb_nodes ; j++) { //&& print
                //System.out.printf("\tsection Node %d\n", j);
                for (int i = 0; i < nb_data; i++) {
                    if (transferHeights[j][i].getValue() == 1) {
                        //System.out.println("  Data " + i + " transferred to " + j + " at " + transferTasks[j][i].getStart().getValue() + " (dur: " + transferTasks[j][i].getDuration().getValue() + ")");
                        //System.out.printf("\t\tData %d : done, %d, %d\n", i, transferTasks[j][i].getStart().getValue(), transferTasks[j][i].getEnd().getValue());
                        int[] wl = works[i];
                        for (int k = 0; k < wl.length; k++) {
                            if (jobNodes[i][k].isInstantiatedTo(j)) {
                                //System.out.printf("     Job %d from %d executed at %d (dur: %d) on node %d\n",
                                //        k, i, jobStarts[i][k].getValue(), jobDurations[i][k].getValue(), j);
                                //System.out.printf("\t\tJob %d-%d : active, %d, %d\n", i, k, jobStarts[i][k].getValue(), jobEnds[i][k].getValue());
                                WorkConfig tmp_work = new WorkConfig(k, i,jobStarts[i][k].getValue(),jobEnds[i][k].getValue(),j);
                                worksList.add(tmp_work);
                            }
                        }

                        TransferConfig tmp_transfer = new TransferConfig(i, transferTasks[j][i].getStart().getValue(), transferTasks[j][i].getEnd().getValue(), j);
                        transfersList.add(tmp_transfer);
                    }
                }

                // TODO
                /*for (Main.WorkEntry we : workTasks) {
                    if (we.nodeIndex == j && we.height.getValue() == 1) {
                        //System.out.println("  Work " + we.workIndex + " of Data " + we.dataIndex+ " processed from " + we.task.getStart() + " to " + we.task.getDuration());

                        WorkConfig tmp_work = new WorkConfig(we.workIndex, we.dataIndex, we.task.getStart().getValue(), we.task.getEnd().getValue(), j);
                        worksList.add(tmp_work);
                    }*/

            }

            //System.out.println();
            //System.out.printf("Objectives = %d -- %d\n", objectives[0].getValue(), objectives[1].getValue());
            for (int i = 0; i < nb_data && print; i++) {
                System.out.printf("Data %d --> max : %d\n", i, Arrays.stream(jobEnds[i]).mapToInt(IntVar::getValue).max().getAsInt() - starting_times[i]);
            }
            System.out.printf("M%d: SumFlow %d; MaxFlow %d; Nb Transfers %d ;Time %.2f; Num Sol %d; \n", pos, objectives[0].getValue(), objectives[1].getValue(), transfersList.size(), solver.getTimeCount(), solver.getSolutionCount());

            result.sumFlowTime = objectives[0].getValue();
            result.maxFlowTime = objectives[1].getValue();
            result.nbTransfers = transfersList.size();
            result.solveTimeSec = solver.getTimeCount();
            result.nbSolutionsFound = solver.getSolutionCount();

            if (outputDir != null) {
                try {
                    writeWorkConfigCSV(worksList, outputDir + "/works_exec_solution.csv");
                    writeTransferConfigCSV(transfersList, outputDir + "/transfers_solution.csv");
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

        });
        //solver.findLexOptimalSolution(objectives, false);

        if (solve) {
            solver.findOptimalSolution(objectives[0], false);
            if (!result.foundSolution) {
                System.out.println("No solution found");
            }
            result.provenOptimal = result.foundSolution && solver.isObjectiveOptimal();
        } else {
            model.setObjective(false, objectives[0]);
            models[pos] = model;
        }

        return result;
    }

    static void setLNS(Solver solver, INeighbor neighbor, ICounter restartCounter) {
        MyMoveLNS lns = new MyMoveLNS(solver.getMove(), neighbor, restartCounter);
        solver.setMove(lns);
    }

    private static IntVar[] decisionVariables(int nb_nodes, int nb_data, int[][] works, IntVar[][] jobNodes, IntVar[][] jobStarts, BoolVar[][] transferHeights, Task[][] transferTasks) {
        List<IntVar> vars = new ArrayList<>();
        for (int i = 0; i < nb_data; i++) {
            for (int k = 0; k < works[i].length; k++) {
                vars.add(jobNodes[i][k]);
                vars.add(jobStarts[i][k]);
            }
            for (int k = 0; k < works[i].length; k++) {
                //vars.add(jobStarts[i][k]);
            }
        }
        for (int i = 0; i < nb_data; i++) {
            for (int j = 0; j < nb_nodes; j++) {
                vars.add(transferHeights[j][i]);
                vars.add(transferTasks[j][i].getStart());
            }
        }
        IntVar[] decisionVars = vars.toArray(new IntVar[0]);
        return decisionVars;
    }

    private static void hints(int nb_nodes, int nb_data, int[] data_sizes, int[][] works, double[] cpus, Solver solver, IntVar[][] jobNodes) {
        ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);
        int[] didx = ArrayUtils.array(0, nb_data - 1);
        sorter.sort(didx, nb_data, (i, j) -> {
            int diff = data_sizes[j] - data_sizes[i];
            if (diff == 0) {
                diff = works[j].length * works[j][0] - works[i].length * works[i][0];
            }
            return diff;
        });
        sorter = new ArraySort<>(nb_nodes, false, true);
        //TODO: there's a problem here that I don't understand
        int[] cidx = ArrayUtils.array(0, nb_nodes - 1);
        sorter.sort(cidx, nb_nodes, (i, j) -> (int) ((cpus[i] - cpus[j]) * 1000));
        int j = 0;
        for (int i = 0; i < nb_data; i++) {
            if(j==nb_nodes){
                j = 0;
            }
            int ii = cidx[j];
            int value = 0;
            int f = Math.ceilDiv(works[i].length, nb_nodes);
            //2;
            int next = Math.ceilDiv(works[i].length, f);
            for (int k = 0; k < works[i].length; k++) {
                //int ii = cidx[Math.floorDiv(k, wpn)];
                //int ii = cidx[(i + value) % nb_nodes];
                solver.addHint(jobNodes[i][k], ii);
                if (k == next) {
                    //f *= 2;
                    next += Math.max(1, Math.ceilDiv(works[i].length, f));
                    value++;
                }
            }
            j++;
        }
    }

    private static void hints2(int nb_nodes, int nb_data, int[] data_sizes, int[][] works, double[] cpus, Solver solver, IntVar[][] jobNodes) {

        // ---- Thresholds to classify "large" vs "light" ----
        int medianData = median(data_sizes);
        int[] workLoads = new int[nb_data];
        for (int i = 0; i < nb_data; i++) {
            workLoads[i] = works[i].length * works[i][0];
        }
        int medianWork = median(workLoads);

        // ---- Classify nodes into 4 categories ----
        // Nodes are sorted by cpus[], the intervals are based on NB_NODES
        // Cat 0: [0,     0.24*N)  → CPU++ / BW--
        // Cat 1: [0.24N, 0.50N)  → CPU-- / BW--
        // Cat 2: [0.50N, 0.74N)  → CPU++ / BW++
        // Cat 3: [0.74N, N)      → CPU-- / BW++

        // Sort nodes by decreasing CPU for a consistent order
        ArraySort<?> nodeSorter = new ArraySort<>(nb_nodes, false, true);
        int[] cidx = ArrayUtils.array(0, nb_nodes - 1);
        nodeSorter.sort(cidx, nb_nodes, (i, j) -> (int) ((cpus[j] - cpus[i]) * 1000)); // decreasing

        // Category intervals
        int[] catStart = new int[4];
        int[] catEnd   = new int[4];
        double[] sizes = {0.24, 0.26, 0.24, 0.26};
        double cumul = 0;
        for (int c = 0; c < 4; c++) {
            catStart[c] = (int) Math.round(cumul * nb_nodes);
            cumul += sizes[c];
            catEnd[c]   = (int) Math.round(cumul * nb_nodes);
        }

        // Per-category cursors (round-robin within each category)
        int[] catCursors = new int[4];
        for (int c = 0; c < 4; c++) catCursors[c] = catStart[c];

        // Helper function: next node in a category (round-robin)
        // Inlined below

        // ---- Sort data: largest first ----
        ArraySort<?> dataSorter = new ArraySort<>(nb_data, false, true);
        int[] didx = ArrayUtils.array(0, nb_data - 1);
        dataSorter.sort(didx, nb_data, (i, j) -> {
            int diff = data_sizes[j] - data_sizes[i];
            if (diff == 0) diff = workLoads[j] - workLoads[i];
            return diff;
        });

        // ---- Dispatch ----
        for (int i = 0; i < nb_data; i++) {
            int di = didx[i];
            boolean bigData = data_sizes[di] >= medianData;
            boolean bigWork = workLoads[di]  >= medianWork;

            // Pick the category from the matrix
            int targetCat;
            if      ( bigData &&  bigWork) targetCat = 2; // CPU++ BW++
            else if ( bigData && !bigWork) targetCat = 3; // CPU-- BW++
            else if (!bigData &&  bigWork) targetCat = 0; // CPU++ BW--
            else                           targetCat = 1; // CPU-- BW-- (light)

            // Target node via round-robin within the category
            int cursor = catCursors[targetCat];
            int nodeIdx = cidx[cursor];

            // Advance the cursor (round-robin within the interval)
            catCursors[targetCat]++;
            if (catCursors[targetCat] >= catEnd[targetCat]) {
                catCursors[targetCat] = catStart[targetCat];
            }

            // Dispatch every work item of the job onto this node
            // (same spread logic as the original if you want to distribute across multiple nodes)
            for (int k = 0; k < works[di].length; k++) {

                solver.addHint(jobNodes[di][k], nodeIdx);
            }
        }
    }

    private static void hints3(int nb_nodes, int nb_data, int[] data_sizes,
                               int[][] works, double[] cpus, int[] bandwidths,
                               Solver solver, IntVar[][] jobNodes) {

        double minCpu = Arrays.stream(cpus).min().getAsDouble();
        int minBw = Arrays.stream(bandwidths).min().getAsInt();

        // SEVEN: sort jobs by increasing estimated cost (shortest first)
        int[] didx = ArrayUtils.array(0, nb_data - 1);
        ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);
        sorter.sort(didx, nb_data, (i, j) -> {
            double costI = (double) data_sizes[i] * minBw
                    + (double) works[i].length * works[i][0] * minCpu;
            double costJ = (double) data_sizes[j] * minBw
                    + (double) works[j].length * works[j][0] * minCpu;
            return Double.compare(costI, costJ);
        });

        // Current estimated load per node
        double[] nodeLoad = new double[nb_nodes];

        for (int i = 0; i < nb_data; i++) {
            int di = didx[i];
            int workDuration = works[di][0];
            int dataSize = data_sizes[di];

            for (int k = 0; k < works[di].length; k++) {

                int bestNode = -1;
                double bestScore = Double.MAX_VALUE;

                for (int n = 0; n < nb_nodes; n++) {
                    double transferCost = (double) dataSize * bandwidths[n];
                    double execCost     = (double) workDuration * cpus[n];
                    double completionTime = nodeLoad[n] + transferCost + execCost;

                    if (completionTime < bestScore) {
                        bestScore = completionTime;
                        bestNode = n;
                    }
                }

                solver.addHint(jobNodes[di][k], bestNode);
                nodeLoad[bestNode] += (double) dataSize * bandwidths[bestNode]
                        + (double) workDuration * cpus[bestNode];
            }
        }
    }
    // Helper: median of an array
    private static int median(int[] arr) {
        int[] sorted = arr.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static INeighbor getNeighbor0(int nb_data, int[][] works, int[] starting_times, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
        return new INeighbor() {
            @Override
            public void recordSolution() {
            }

            @Override
            public void fixSomeVariables() throws ContradictionException {
            }

            @Override
            public void loadFromSolution(Solution solution) {
            }

            @Override
            public void restrictLess() {
            }
        };
    }

    private static INeighbor getNeighbor1(int nb_data, int[][] works, int[] starting_times, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
        return new INeighbor() {
            int[][] jn;
            int[][] sn;
            int[] maxs;
            int[] imaxs;
            int lim = 0;
            int loops = 0;
            final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


            @Override
            public void recordSolution() {
                jn = new int[nb_data][];
                sn = new int[nb_data][];
                maxs = new int[nb_data];
                imaxs = new int[nb_data];
                for (int i = 0; i < nb_data; i++) {
                    jn[i] = new int[works[i].length];
                    sn[i] = new int[works[i].length];
                    for (int k = 0; k < works[i].length; k++) {
                        jn[i][k] = jobNodes[i][k].getValue();
                        sn[i][k] = jobStarts[i][k].getValue();
                    }
                    final int ii = i;
                    maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - starting_times[ii]).max().getAsInt();
                    imaxs[i] = i;
                }
                sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                lim = 0;
                loops = 1;
            }

            @Override
            public void fixSomeVariables() throws ContradictionException {
                for (int i = 0; i < nb_data /*&& loops < 1000*/; i++) {
                    int ii = imaxs[i];
                    for (int k = 0; k < works[ii].length; k++) {
                        if (i == lim) {
                            jobNodes[ii][k].removeValue(jn[ii][k], this);
                        } else {
                            jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                            jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                        }
                    }
                }
            }

            @Override
            public void loadFromSolution(Solution solution) {

            }

            @Override
            public void restrictLess() {
                lim = (lim + 1) % nb_data;
                if (lim == 0) {
                    loops++;
                    //System.out.printf("Loops %d\n", loops);
                }
            }
        };
    }

    private static INeighbor getNeighbor2(int nb_data, int[][] works, int[] starting_times, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
        return new INeighbor() {
            int[][] jn;
            int[][] sn;
            int[] maxs;
            int[] imaxs;
            int data = 0;
            int work = 0;
            final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
            final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


            @Override
            public void recordSolution() {
                jn = new int[nb_data][];
                sn = new int[nb_data][];
                maxs = new int[nb_data];
                imaxs = new int[nb_data];
                mapping.clear();
                for (int i = 0; i < nb_data; i++) {
                    jn[i] = new int[works[i].length];
                    sn[i] = new int[works[i].length];
                    for (int k = 0; k < works[i].length; k++) {
                        jn[i][k] = jobNodes[i][k].getValue();
                        TIntArrayList list = mapping.get(i);
                        if (list == null) {
                            list = new TIntArrayList();
                            mapping.put(i, list);
                        }
                        if (!list.contains(jn[i][k])) {
                            list.add(jn[i][k]);
                        }
                        sn[i][k] = jobStarts[i][k].getValue();
                    }
                    final int ii = i;
                    maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - starting_times[ii]).max().getAsInt();
                    imaxs[i] = i;
                }
                sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                data = 0;
                work = 0;
            }

            @Override
            public void fixSomeVariables() throws ContradictionException {
                boolean move = false;
                for (int i = 0; i < nb_data; i++) {
                    int ii = imaxs[i];
                    TIntArrayList values = mapping.get(ii);
                    if (i == data) {
                        for (int k = 0; k < works[ii].length; k++) {
                            if (jn[ii][k] == values.get(work)) {
                                jobNodes[ii][k].removeValue(jn[ii][k], this);
                            } else {
                                jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                                jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                            }
                        }
                        work++;
                        if (work == values.size()) {
                            move = true;
                        }
                    } else {
                        for (int k = 0; k < works[ii].length; k++) {
                            jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                            jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                        }
                    }
                }
                if (move) {
                    data = (data + 1) % nb_data;
                    work = 0;
                }
            }

            @Override
            public void loadFromSolution(Solution solution) {

            }

            @Override
            public void restrictLess() {
            }
        };
    }

    private static INeighbor getNeighbor3(int nb_data, int[][] works, int[] starting_times, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
        return new INeighbor() {
            int[][] jn;
            int[][] sn;
            int[] maxs;
            int[] imaxs;
            int node = 0;
            final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
            final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


            @Override
            public void recordSolution() {
                jn = new int[nb_data][];
                sn = new int[nb_data][];
                maxs = new int[nb_data];
                imaxs = new int[nb_data];
                mapping.clear();
                for (int i = 0; i < nb_data; i++) {
                    jn[i] = new int[works[i].length];
                    sn[i] = new int[works[i].length];
                    for (int k = 0; k < works[i].length; k++) {
                        jn[i][k] = jobNodes[i][k].getValue();
                        TIntArrayList list = mapping.get(jn[i][k]);
                        if (list == null) {
                            list = new TIntArrayList();
                            mapping.put(jn[i][k], list);
                        }
                        if (!list.contains(i)) {
                            list.add(i);
                        }
                        sn[i][k] = jobStarts[i][k].getValue();
                    }
                    final int ii = i;
                    maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - starting_times[ii]).max().getAsInt();
                    imaxs[i] = i;
                }
                sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                node = 0;
            }

            @Override
            public void fixSomeVariables() throws ContradictionException {
                TIntArrayList datas = mapping.get(mapping.keys()[node]);
                for (int i = 0; i < nb_data; i++) {
                    if (datas.contains(i)) {
                        for (int k = 0; k < works[i].length; k++) {
                            jobNodes[i][k].removeValue(jn[i][k], this);
                        }
                    } else {
                        for (int k = 0; k < works[i].length; k++) {
                            jobNodes[i][k].instantiateTo(jn[i][k], this);
                            //jobStarts[i][k].instantiateTo(sn[i][k], this);
                        }
                    }
                }
                node = (node + 1) % mapping.keys().length;
            }

            @Override
            public void loadFromSolution(Solution solution) {

            }

            @Override
            public void restrictLess() {
            }
        };
    }

    private static INeighbor getNeighbor1bis(int nb_data, int[][] works, int[] starting_times, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
        return new INeighbor() {
            int[][] jn;
            int[][] sn;
            int[] maxs;
            int[] imaxs;
            int lim = 0;
            final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


            @Override
            public void recordSolution() {
                jn = new int[nb_data][];
                sn = new int[nb_data][];
                maxs = new int[nb_data];
                imaxs = new int[nb_data];
                for (int i = 0; i < nb_data; i++) {
                    jn[i] = new int[works[i].length];
                    sn[i] = new int[works[i].length];
                    for (int k = 0; k < works[i].length; k++) {
                        jn[i][k] = jobNodes[i][k].getValue();
                        sn[i][k] = jobStarts[i][k].getValue();
                    }
                    final int ii = i;
                    maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - starting_times[ii]).max().getAsInt();
                    imaxs[i] = i;
                }
                sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                lim = 0;
            }

            @Override
            public void fixSomeVariables() throws ContradictionException {
                //System.out.printf("1bis : %d\n", lim);
                for (int i = 0; i < nb_data; i++) {
                    int ii = imaxs[i];
                    for (int k = 0; k < works[ii].length; k++) {
                        if (i != lim) {
                            jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                            //jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                        }
                    }
                }
            }

            @Override
            public void loadFromSolution(Solution solution) {

            }

            @Override
            public void restrictLess() {
                lim = (lim + 1) % nb_data;
            }
        };
    }

    private static INeighbor getNeighbor2bis(int nb_data, int[][] works, int[] starting_times, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
        return new INeighbor() {
            int[][] jn;
            int[][] sn;
            int[] maxs;
            int[] imaxs;
            int data = 0;
            int work = 0;
            final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
            final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


            @Override
            public void recordSolution() {
                jn = new int[nb_data][];
                sn = new int[nb_data][];
                maxs = new int[nb_data];
                imaxs = new int[nb_data];
                mapping.clear();
                for (int i = 0; i < nb_data; i++) {
                    jn[i] = new int[works[i].length];
                    sn[i] = new int[works[i].length];
                    for (int k = 0; k < works[i].length; k++) {
                        jn[i][k] = jobNodes[i][k].getValue();
                        TIntArrayList list = mapping.get(i);
                        if (list == null) {
                            list = new TIntArrayList();
                            mapping.put(i, list);
                        }
                        if (!list.contains(jn[i][k])) {
                            list.add(jn[i][k]);
                        }
                        sn[i][k] = jobStarts[i][k].getValue();
                    }
                    final int ii = i;
                    maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - starting_times[ii]).max().getAsInt();
                    imaxs[i] = i;
                }
                sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                data = 0;
                work = 0;
            }

            @Override
            public void fixSomeVariables() throws ContradictionException {
                //System.out.printf("2bis : %d - %d\n", imaxs[data], work);
                boolean move = false;
                for (int i = 0; i < nb_data; i++) {
                    int ii = imaxs[i];
                    TIntArrayList values = mapping.get(ii);
                    if (i == data) {
                        for (int k = 0; k < works[ii].length; k++) {
                            if (jn[ii][k] != values.get(work)) {
                                jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                                //jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                            }
                        }
                        work++;
                        if (work == values.size()) {
                            move = true;
                        }
                    } else {
                        for (int k = 0; k < works[ii].length; k++) {
                            jobNodes[ii][k].instantiateTo(jn[ii][k], this);
                            //jobStarts[ii][k].instantiateTo(sn[ii][k], this);
                        }
                    }
                }
                if (move) {
                    data = (data + 1) % nb_data;
                    work = 0;
                }
            }

            @Override
            public void loadFromSolution(Solution solution) {

            }

            @Override
            public void restrictLess() {
            }
        };
    }

    private static INeighbor getNeighbor3bis(int nb_data, int[][] works, int[] starting_times, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
        return new INeighbor() {
            int[][] jn;
            int[][] sn;
            int[] maxs;
            int[] imaxs;
            int node = 0;
            final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
            final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


            @Override
            public void recordSolution() {
                jn = new int[nb_data][];
                sn = new int[nb_data][];
                maxs = new int[nb_data];
                imaxs = new int[nb_data];
                mapping.clear();
                for (int i = 0; i < nb_data; i++) {
                    jn[i] = new int[works[i].length];
                    sn[i] = new int[works[i].length];
                    for (int k = 0; k < works[i].length; k++) {
                        jn[i][k] = jobNodes[i][k].getValue();
                        TIntArrayList list = mapping.get(jn[i][k]);
                        if (list == null) {
                            list = new TIntArrayList();
                            mapping.put(jn[i][k], list);
                        }
                        if (!list.contains(i)) {
                            list.add(i);
                        }
                        sn[i][k] = jobStarts[i][k].getValue();
                    }
                    final int ii = i;
                    maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - starting_times[ii]).max().getAsInt();
                    imaxs[i] = i;
                }
                sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                node = 0;
            }

            @Override
            public void fixSomeVariables() throws ContradictionException {
                //System.out.printf("3bis : %d\n", mapping.keys()[node]);
                TIntArrayList datas = mapping.get(mapping.keys()[node]);
                for (int i = 0; i < nb_data; i++) {
                    if (!datas.contains(i)) {
                        for (int k = 0; k < works[i].length; k++) {
                            jobNodes[i][k].instantiateTo(jn[i][k], this);
                            //jobStarts[i][k].instantiateTo(sn[i][k], this);
                        }
                    }
                }
                node = (node + 1) % mapping.keys().length;
            }

            @Override
            public void loadFromSolution(Solution solution) {

            }

            @Override
            public void restrictLess() {
            }
        };
    }

    private static INeighbor getNeighbor4bis(int nb_data, int[][] works, int[] starting_times, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
        return new INeighbor() {
            int[][] jn;
            int[][] sn;
            int[] maxs;
            int[] imaxs;
            int node1 = 0;
            int node2 = 0;
            final TIntObjectHashMap<TIntArrayList> mapping = new TIntObjectHashMap<>();
            final ArraySort<?> sorter = new ArraySort<>(nb_data, false, true);


            @Override
            public void recordSolution() {
                jn = new int[nb_data][];
                sn = new int[nb_data][];
                maxs = new int[nb_data];
                imaxs = new int[nb_data];
                mapping.clear();
                for (int i = 0; i < nb_data; i++) {
                    jn[i] = new int[works[i].length];
                    sn[i] = new int[works[i].length];
                    for (int k = 0; k < works[i].length; k++) {
                        jn[i][k] = jobNodes[i][k].getValue();
                        TIntArrayList list = mapping.get(jn[i][k]);
                        if (list == null) {
                            list = new TIntArrayList();
                            mapping.put(jn[i][k], list);
                        }
                        if (!list.contains(i)) {
                            list.add(i);
                        }
                        sn[i][k] = jobStarts[i][k].getValue();
                    }
                    final int ii = i;
                    maxs[i] = Arrays.stream(jobEnds[i]).mapToInt(v -> v.getValue() - starting_times[ii]).max().getAsInt();
                    imaxs[i] = i;
                }
                sorter.sort(imaxs, nb_data, (i, j) -> maxs[j] - maxs[i]);
                node1 = 0;
                node2 = 0;
            }

            @Override
            public void fixSomeVariables() throws ContradictionException {
                //System.out.printf("3bis : %d\n", mapping.keys()[node]);
                TIntArrayList datas1 = mapping.get(mapping.keys()[node1]);
                TIntArrayList datas2 = mapping.get(mapping.keys()[node2]);
                for (int i = 0; i < nb_data; i++) {
                    if (!datas1.contains(i) && !datas2.contains(i)) {
                        for (int k = 0; k < works[i].length; k++) {
                            jobNodes[i][k].instantiateTo(jn[i][k], this);
                            //jobStarts[i][k].instantiateTo(sn[i][k], this);
                        }
                    }
                }
                node2 = (node2 + 1) % mapping.keys().length;
                if (node2 == 0) {
                    node1 = (node1 + 1) % mapping.keys().length;
                }
            }

            @Override
            public void loadFromSolution(Solution solution) {

            }

            @Override
            public void restrictLess() {
            }
        };
    }

    private static INeighbor getNeighbor4(int nb_data, int[][] works, int[] starting_times, IntVar[][] jobNodes, IntVar[][] jobStarts, IntVar[][] jobEnds) {
        return new INeighbor() {
            int[][] jn;
            int[][] sn;
            List<Integer> fixed = IntStream.range(0, nb_data).boxed().collect(Collectors.toList());
            java.util.Random rnd = new java.util.Random(42);
            int nbFixed;
            int round;

            @Override
            public void recordSolution() {
                jn = new int[nb_data][];
                sn = new int[nb_data][];
                for (int i = 0; i < nb_data; i++) {
                    jn[i] = new int[works[i].length];
                    sn[i] = new int[works[i].length];
                    for (int k = 0; k < works[i].length; k++) {
                        jn[i][k] = jobNodes[i][k].getValue();
                        sn[i][k] = jobStarts[i][k].getValue();
                    }
                }
                nbFixed = nb_data - 1;
                round = 1;
            }

            @Override
            public void fixSomeVariables() throws ContradictionException {
                Collections.shuffle(fixed, rnd);
                for (int i = 0; i < nbFixed; i++) {
                    for (int k = 0; k < works[i].length; k++) {
                        jobNodes[i][k].instantiateTo(jn[i][k], this);
                        jobStarts[i][k].instantiateTo(sn[i][k], this);
                    }
                }
                round++;
            }

            @Override
            public void loadFromSolution(Solution solution) {

            }

            @Override
            public void restrictLess() {
                if (round % 400 == 0) {
                    nbFixed--;
                }
            }
        };
    }

}