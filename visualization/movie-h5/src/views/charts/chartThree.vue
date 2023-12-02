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
      let xAxis = data.map(item => item.regParam)
      let rmse = data.map(item => item.RMSE)
      let rank = data.map(item => item.rank)
      let option = {
        title: {
          text: 'RMSE vs regParam for different rank',
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
              color: '#9dd2e7'
            },
            lineStyle: {
              color: '#9dd2e7'
            },
            itemStyle: {
              color: '#9dd2e7'
            },
            emphasis: {
              focus: 'series'
            },
            yAxisIndex: 0,
            data: rmse,
            zIndex: 1
          },
          {
            name: 'Email',
            type: 'line',
            yAxisIndex: 1,
            data: rank,
            zIndex: 3,
            lineStyle: {
              color: '#889bd8'
            },
            itemStyle: {
              color: '#889bd8'
            },
          },

        ]
      };
      this.chart.setOption(option);
    },
  },
};
</script>