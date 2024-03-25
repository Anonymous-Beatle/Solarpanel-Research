data_str = '<div class="" style="position: absolute; display: block; border-style: solid; white-space: nowrap; z-index: 9999999; will-change: transform; box-shadow: rgba(0, 0, 0, 0.2) 1px 2px 10px; transition: opacity 0.2s cubic-bezier(0.23, 1, 0.32, 1) 0s, visibility 0.2s cubic-bezier(0.23, 1, 0.32, 1) 0s, transform 0.4s cubic-bezier(0.23, 1, 0.32, 1) 0s; background-color: rgb(255, 255, 255); border-width: 1px; border-radius: 4px; color: rgb(102, 102, 102); font: 14px / 21px &quot;Microsoft YaHei&quot;; padding: 10px; top: 0px; left: 0px; transform: translate3d(355px, 142px, 0px); border-color: rgb(255, 255, 255); pointer-events: none;">2023-06-24 11:35<br><span style="display:inline-block;margin-right:4px;border-radius:10px;width:10px;height:10px;background-color:#18CF87;"></span>PV output<span style="float:right;font-weight:bold;margin-left:20px;">2.584</span><br></div>'

print(data_str.split("pointer-events:")[1].split("<")[0].split(">")[1])



my_yield = '<span style="float:right;font-weight:bold;margin-left:20px;">2.584</span>'

print(my_yield.split(">")[1].split("<")[0])