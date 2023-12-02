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
      let xAxis = data.map(item => item.MaxBins)
      let rmse = data.map(item => item.MAE)
      let rank = data.map(item => item.NumTrees)
      let option = {
        title: {
          text: 'MAE vs MaxBins for different rank',
          left: 10
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: [
          {
            type: 'category',
            boundaryGap: false,
            data: xAxis
          }
        ],
        yAxis: [
          {
            type: 'value',
          },
          {
            type: 'value',
            max: function(value) {return value.max + 5;}
          }
        ],
        series: [
          {
            name: 'Email',
            type: 'line',
            stack: 'Total',
            areaStyle: {
              color: '#fc8452'
            },
            lineStyle: {
              color: '#fc8452'
            },
            itemStyle: {
              color: '#fc8452'
            },
            emphasis: {
              focus: 'series'
            },
            yAxisIndex: 0,
            data: rmse
          },
          {
            name: 'Email',
            type: 'line',
            yAxisIndex: 1,
            data: rank,
            lineStyle: {
              color: '#5470c6'
            },
            itemStyle: {
              color: '#5470c6'
            },
          },

        ]
      };
      this.chart.setOption(option);
    },
  },
};
</script>