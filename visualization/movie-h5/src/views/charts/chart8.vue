<template>
  <div class="chart" style="width: 100%; height: 230px"></div>
</template>
<script>
import * as echarts from "echarts";
require("echarts/theme/macarons"); // echarts theme
import resize from "./mixins/resize";
export default {
  mixins: [resize],
  data() {
    return {
      chart: null,
    };
  },
  mounted() {
    this.$nextTick(() => {
      this.chart = echarts.init(this.$el, "macarons");
      this.initChart();
    });
  },
  beforeDestroy() {
    if (!this.chart) {
      return
    }
    this.chart.dispose()
    this.chart = null
  },
  methods: {
    initChart() {
      let data = require('@/utils/rf.json')
      console.log(data)
      let rank5 = []
      let rank10 = []
      let rank20 = []
      data.forEach(item => {
        if (item.NumTrees == 5) rank5.push(Number(item['ExecutionTime']))
        if (item.NumTrees == 10) rank10.push(Number(item['ExecutionTime']))
        if (item.NumTrees == 20) rank20.push(Number(item['ExecutionTime']))
      })
      let max = Math.max(rank5.length, rank10.length, rank20.length)
      let option = {
        title: {
          text: 'ExecutionTime for different NumTrees',
          left: 10
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        legend: {
          show: false
        },
        xAxis: { type: 'category', data: ['5', '10', '20'] },
        yAxis: { type: 'value' },
        series: [{
          type: 'bar',
          data: rank5
        }, {
          type: 'bar',
          data: rank10
        }, {
          type: 'bar',
          data: rank20
        }]
      };
      this.chart.setOption(option);
    },
  },
};
</script>