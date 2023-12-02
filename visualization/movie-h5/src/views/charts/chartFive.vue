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
      let data = require('@/utils/als.json')
      console.log(data)
      let rank10 = []
      let rank20 = []
      data.forEach(item => {
        if (item.rank == 10) rank10.push(Number(item['Training Time']))
        if (item.rank == 20) rank20.push(Number(item['Training Time']))
      })
      console.log(1111, rank10, rank20)
      let option = {
        title: {
          text: 'Training Time for different rank',
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
        dataset: {
          source: [
            ['10', ...rank10],
            ['20', ...rank20],
          ]
        },
        xAxis: { type: 'category' },
        yAxis: {},
        series: [{ type: 'bar', barWidth: '20px' }, { type: 'bar', barWidth: '20px' }, { type: 'bar', barWidth: '20px' }, { type: 'bar', barWidth: '20px' }]
      };
      this.chart.setOption(option);
    },
  },
};
</script>